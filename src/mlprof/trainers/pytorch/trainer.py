"""
ddp/trainer.py
"""
from __future__ import absolute_import, annotations, division, print_function
import time
import os
from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from contextlib import nullcontext
# from rich import print_json
from pathlib import Path
# from dataclasses import asdict
import json
import torch
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import wandb
import numpy as np

try:
    import horovod.torch as hvd
except (ImportError, ModuleNotFoundError):
    hvd = None

from mlprof.trainers.trainer import BaseTrainer
from mlprof.configs import (
    DATA_DIR,
    # HERE,
    ExperimentConfig,
    ConvNetworkConfig,
    # SYNONYMS
)
from mlprof.network.pytorch.network import Net
# from mlprof.utils.pylogger import get_pylogger
from mlprof.utils.dist import setup_torch_distributed

# log = get_pylogger(__name__)
from mlprof import get_logger
log = get_logger(__name__)


def metric_average(val: torch.Tensor, size: int = 1):
    try:
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
    except Exception as exc:
        log.exception(exc)

    return val / size


def load_ds_config(fpath: os.PathLike) -> dict:
    ds_config_path = Path(fpath)
    with ds_config_path.open('r') as f:
        ds_config = json.load(f)

    return ds_config


class Trainer(BaseTrainer):
    def __init__(
            self,
            config: DictConfig,
            model: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.cfg = config
        self.config: ExperimentConfig = (
            # config if isinstance(config, ExperimentConfig)
            instantiate(config)
        )
        dsetup = setup_torch_distributed(self.config.backend)
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.world_size = dsetup['size']
        self.rank = dsetup['rank']
        self.local_rank = dsetup['local_rank']
        self._is_rank_0 = (self.local_rank == 0 and self.rank == 0)
        if torch.cuda.is_available():
            self.local_size = torch.cuda.device_count()
            self._ngpus = self.world_size
            # self._ngpus = self.world_size * torch.cuda.device_count()

        # self.wbrun = self.setup_wandb()
        self.wbrun = wandb.run

        # if scaler is None:
        #     self.scaler = None
        self.scaler = None
        # if (
        #         scaler is None
        #         # and self.config.backend == 'DDP'
        #         and self.config.backend.lower() not in ['ds', 'deepspeed']
        # ):
        #     self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        assert isinstance(self.config, ExperimentConfig)
        self._global_step = 0

        self.loss_fn = nn.CrossEntropyLoss()

        self.data = self.setup_data()
        self.model = (
            model if model is not None and isinstance(model, nn.Module)
            else self.build_model(self.config.network)
        )
        if optimizer is None:
            self._optimizer = self.build_optimizer(
                model=self.model,
                lr_init=self.config.trainer.lr_init
            )

        self.dtype = torch.get_default_dtype()
        if torch.cuda.is_available():
            self.model.cuda()
            # if self.config.autocast:
            #     self.dtype = torch.get_autocast_gpu_dtype()
            #     self.model.to(self.dtype)
            # self.loss_fn = self.loss_fn.to(self.dtype)

        # self.wbrun = wbrun
        self.num_parameters = self.count_parameters(self.model)
        if self.wbrun is not None:
            log.warning(f'Caught wandb.run from: {self.rank}')
            self.wbrun.watch(
                self.model,
                log='all',
                criterion=self.loss_fn,
                log_graph=True,
                log_freq=self.config.trainer.logfreq,
            )

        self.use_fp16 = False
        self.ds_config = None
        self.model_engine = self.model
        if self.config.backend == 'DDP':
            self.model_engine = DDP(self.model)
            self.optimizer: torch.optim.Optimizer = self._optimizer

        elif self.config.backend.lower() in ['ds', 'deepspeed']:
            import deepspeed
            self.ds_config = self.prepare_ds_config()
            params = filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            )
            engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=params,  # type: ignore
                # optimizer=self._optimizer,
                config=self.ds_config
            )
            self.model_engine = engine
            assert optimizer is not None
            self.optimizer: torch.optim.Optimizer = optimizer
            self.use_fp16 = self.model_engine.fp16_enabled()
            if self.use_fp16:
                self.dtype = torch.half
            if self.wbrun is not None:
                self.wbrun.config['ds_config'] = self.ds_config

        elif self.config.backend.lower() in ['hvd', 'horovod']:
            import horovod.torch as hvd
            # self.model_engine = self.model
            compression = (
                hvd.Compression.fp16
                if self.config.compression == 'fp16'
                else hvd.Compression.none
            )
            self.optimizer = hvd.DistributedOptimizer(
                self._optimizer,
                named_parameters=self.model.named_parameters(),
                compression=compression  # type:ignore
            )
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.world_size = hvd.size()
            self._ngpus = self.world_size

        else:
            # self.model_engine = None
            # self.optimizer = self._optimizer
            raise ValueError(
                f'Unexpected value for `config.backend`: {self.config.backend}'
            )

    def count_parameters(self, model: Optional[nn.Module] = None) -> int:
        """Count the total number of parameters in `model`."""
        model = self.model if model is None else model
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        log.info(f'num_params in model: {num_params}')
        if wandb.run is not None:
            wandb.run.config['NUM_PARAMS'] = num_params
        return num_params

    def prepare_ds_config(self) -> dict:
        if self.config.backend.lower() not in ['ds', 'deepspeed']:
            return {}

        ds_config = self.config.load_ds_config(self.config.ds_config_path)
        ds_config |= {
            'wandb': {
                'project': 'mlprof',
                # 'group': f'{self.config.framework}/{self.config.backend}',
            }
        }
        # ds_config['wandb'].update({
        #     "project": pname,
        #     "group": f'{self.config.framework}/{self.config.backend}',
        # })
        # self.ds_config.update({
        #     'tensorboard': {
        #         'enabled': True,
        #         'output_path': Path(os.getcwd()),
        #     },
        #     'csv_monitor': {
        #         'enabled': True,
        #         'output_path': Path(os.getcwd()).joinpath('csv_monitor'),
        #     }
        # })
        ds_config |= {
            'gradient_accumulation_steps': 1,
            'train_micro_batch_size_per_gpu': 1,
        }
        ds_config['train_batch_size'] = (
            self._ngpus
            * ds_config['gradient_accumulation_steps']
            * ds_config['train_micro_batch_size_per_gpu']
        )
        # if ds_config['scheduler'].get('params', None) is not None:
        #     ds_config['scheduler']['params'].update({
        #         'warmup_num_steps': self.config.steps.nepoch,
        #         'total_num_steps': (
        #             self.config.steps.nera * self.config.steps.nepoch
        #         )
        #     })
        return ds_config

    def setup_wandb(self):
        wbrun = None
        wbcfg = self.config.wandb
        if self._is_rank_0 and wbcfg is not None:
            from mlprof.common import setup_wandb
            setup_wandb(self.cfg)
            wbrun = wandb.run
            assert wbrun is not None
            # import wandb
            # try:
            #     from wandb.util import generate_id  # type:ignore
            #     run_id = generate_id()
            # except (ImportError, ModuleNotFoundError) as e:
            #     log.exception(e)
            #     import uuid
            #     run_id = str(uuid.uuid1())
            # # wandb.tensorboard.patch(root_logdir=os.getcwd())
            # wbrun = wandb.init(
            #     dir=os.getcwd(),
            #     id=run_id,
            #     mode='online',
            #     resume='allow',
            #     save_code=True,
            #     project=self.config.wandb.setup.project,
            #     # config_exclude_keys=['_target_'],
            # )
            # assert wbrun is not None
            # log.info(f'wandb.name: {wbrun.name}')
            # log.info(f'wandb.url: {wbrun.url}')
            # log.info(f'directory: {os.getcwd()}')
            # assert wbrun is not None and wbrun is wandb.run
            # wbrun.log_code(HERE.as_posix())
            # # cfg = asdict(self.config)
            # wbrun.config.update(
            #     # OmegaConf.to_container(cfg, resolve=True)
            #     asdict(self.config)
            #     # self.config.asdict()
            # )
            # # wbrun.config['backend'] = sel
            # wbrun.config['backend'] = SYNONYMS[self.config.backend]
            # wbrun.config['framework'] = SYNONYMS[self.config.framework]
            # wbrun.config['world_size'] = self.world_size
            # wbrun.config['run_id'] = run_id
            # wbrun.config['run_name'] = wbrun.name
            # wbrun.config['logdir'] = os.getcwd()
            # wbrun.config['ngpus'] = self._ngpus
            # wbrun.config['device'] = self.device
            LOGFILE = os.environ.get('LOGFILE', None)
            NGPUS = os.environ.get('NGPUS', None)
            NRANKS = os.environ.get('NRANKS', None)
            NGPU_PER_RANK = os.environ.get('NGPU_PER_RANK', None)
            if LOGFILE is not None:
                wbrun.config['logfile'] = LOGFILE
                wbrun.save(LOGFILE, policy='end')
            if NRANKS is not None:
                wbrun.config['NRANKS'] = NRANKS
                # log.info(f'NRANKS: {NRANKS}')
            if NGPU_PER_RANK is not None:
                wbrun.config['NGPU_PER_RANK'] = NGPU_PER_RANK
                # log.info(f'NGPU_PER_RANK: {NGPU_PER_RANK}')
            if NGPUS is not None:
                log.info(f'world_size: {self.world_size}')
                log.info(f'NGPUS: {NGPUS}')
                wbrun.config['NGPUS'] = NGPUS
            import mlprof.utils.dist as dist
            from mlprof.common import ENV_FILTERS
            wbrun.config['MACHINE'] = dist.get_machine()
            distenv = dist.query_environment()
            wbrun.config['WORLD_SIZE'] = distenv['world_size']
            wbrun.config['RANK'] = distenv['rank']
            environ = {
                k: v for k, v in dict(os.environ).items()
                if (
                    k not in ENV_FILTERS
                    and not k.startswith('_ModuleTable')
                    and not k.startswith('BASH_FUNC_')
                )
            }
            wbrun.config['env'] = environ

        return wbrun

    def build_model(self, net_config: ConvNetworkConfig) -> nn.Module:
        assert net_config is not None
        model = Net(net_config)
        xshape = (1, *self._xshape)
        x = torch.rand((self.config.data.batch_size, *xshape))
        if torch.cuda.is_available():
            model.cuda()
            x = x.cuda()

        _ = model(x)

        return model

    def build_optimizer(
            self,
            lr_init: float,
            model: nn.Module
    ) -> torch.optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr_init)

    def setup_torch(self):
        # torch.manual_seed(self.config.trainer.seed)
        # if self.device == 'gpu':
        # if torch.cuda.is_available():
        #     # DDP: pin GPU to local rank
        #     # torch.cuda.set_device(int(LOCAL_RANK))
        #     torch.cuda.manual_seed(self.config.trainer.seed)
        if (
                self.config.trainer.num_threads is not None
                and isinstance(self.config.trainer.num_threads, int)
                and self.config.trainer.num_threads > 0
        ):
            torch.set_num_threads(self.config.trainer.num_threads)

            log.info('\n'.join([
                'Torch Thread Setup:',
                f' Number of threads: {torch.get_num_threads()}',
            ]))

    def get_mnist_datasets(self) -> dict[str, torch.utils.data.Dataset]:
        train_dataset = (
            datasets.MNIST(
                DATA_DIR.as_posix(),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )
        test_dataset = (
            datasets.MNIST(
                DATA_DIR.as_posix(),
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        self._xshape = [28, 28]
        return {
            'train': train_dataset,
            'test': test_dataset,
        }

    def get_fashionmnist_datasets(self) -> dict[str, torch.utils.data.Dataset]:
        train_dataset = (
            datasets.FashionMNIST(
                DATA_DIR.as_posix(),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )
        test_dataset = (
            datasets.FashionMNIST(
                DATA_DIR.as_posix(),
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        self._xshape = [28, 28]
        return {'train': train_dataset, 'test': test_dataset}

    def get_datasets(self, dset: str) -> dict[str, torch.utils.data.Dataset]:
        assert dset in datasets.__all__
        dset_obj = __import__(f'{datasets}.{dset}')
        assert isinstance(dset_obj, torch.utils.data.Dataset)
        assert callable(dset_obj)
        train_dataset = dset_obj(
            DATA_DIR.as_posix(),
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        test_dataset = dset_obj(
            DATA_DIR.as_posix(),
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        return {
            'train': train_dataset,
            'test': test_dataset,
        }

    def setup_data(
            self,
            datasets: Optional[dict[str, torch.utils.data.Dataset]] = None,
    ):
        kwargs = {}

        if self.device == 'cuda':
            kwargs = {'num_workers': 0, 'pin_memory': True}

        if datasets is None:
            # datasets = self.get_mnist_datasets()
            # if self.config.dataset.lower() == 'fashionmnist':
            if self.config.data.dataset.lower() == 'fashionmnist':
                datasets = self.get_fashionmnist_datasets()
            else:
                datasets = self.get_mnist_datasets()

        assert 'train' in datasets and 'test' in datasets
        train_dataset = datasets['train']
        test_dataset = datasets['test']

        self._xshape = [28, 28]
        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=self.world_size, rank=self.rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.data.batch_size
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.cuda.is_available():
            data = data.to(self.dtype).cuda()
            target = target.to(torch.long).cuda()
            # target = target.to(self.dtype)
            # data, target = data.cuda(), target.cuda()

        self.optimizer.zero_grad()
        # with torch.cuda.amp.autocast():  # type:ignore
        # with torch.autocast(self._device):
        # if self.model_engine is not None:
        # assert self.config.backend.lower() in [
        #     'ds',
        #     'deepspeed',
        #     'DDP',
        # ]
        # i.e. using either DDP or DeepSpeed backends
        probs = self.model_engine(data)
        loss = self.loss_fn(probs, target)
        if self.config.backend.lower() in ['ds', 'deepspeed']:
            self.model_engine.backward(loss)  # type: ignore
            self.model_engine.step()          # type: ignore
        else:
            # if:
            # ctx = torch.autocast(  # type:ignore
            #         device_type=self.device,
            #         dtype=self.dtype
            # )
            # else:
            ctx = nullcontext()

            with ctx:
                probs = self.model(data)
                loss = self.loss_fn(probs, target)
            if self.scaler is not None and isinstance(self.scaler, GradScaler):
                self.scaler.scale(loss).backward()  # type:ignore
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        # if self.scaler is not None and isinstance(self.scaler, GradScaler):
        #     self.scaler.scale(loss).backward()  # type:ignore
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        # loss.backward()
        # self.optimizer.step()

        _, pred = probs.data.max(1)
        acc = (pred == target).sum()
        self._global_step += 1

        return loss, acc

    def has_wbrun(self):
        assert self.wbrun is not None
        return (
            self.rank == 0
            and self.wbrun is not None
            and self.wbrun is wandb.run
        )

    def metric_average(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.backend.lower() == 'ddp':
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif self.config.backend.lower() in ['hvd', 'horovod']:
            if hvd is not None:
                hvd.allreduce_(x)

        return x / self.world_size

    def train_epoch(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        # start = time.time()
        running_acc = torch.tensor(0., dtype=self.dtype)
        running_loss = torch.tensor(0., dtype=self.dtype)
        if torch.cuda.is_available():
            running_acc = running_acc.cuda()
            running_loss = running_loss.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)

        for bidx, (data, target) in enumerate(train_loader):
            t0 = time.time()
            loss, acc = self.train_step(data, target)
            running_acc += acc
            running_loss += loss.item()
            if bidx % self.config.trainer.logfreq == 0 and self.rank == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'step': self._global_step,
                    'dt': time.time() - t0,
                    'batch_acc': acc.item() / self.config.data.batch_size,
                    'batch_loss': loss.item() / self.config.data.batch_size,
                    'acc': running_acc / len(self.data['train']['sampler']),
                    'running_loss': (
                        running_loss / len(self.data['train']['sampler'])
                    ),
                }
                pre = [
                    f'[{self.rank}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.config.trainer.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))
                try:
                    self.wbrun.log(  # type:ignore
                        {f'batch/{key}': val for key, val in metrics.items()}
                    )
                except Exception:
                    pass

        running_loss = running_loss / len(train_sampler)
        running_acc = running_acc / len(train_sampler)
        training_acc = self.metric_average(running_acc)
        loss_avg = self.metric_average(running_loss)
        if self.rank == 0:
            assert (
                self.wbrun is not None
                and self.wbrun is wandb.run  # type:ignore
            )
            self.wbrun.log({'train/loss': loss_avg, 'train/acc': training_acc})
        # if self.config.backend in ['ds', 'deepspeed'] and self.rank == 0:
        #     dscomm.log_summary()

        return {'loss': loss_avg, 'acc': training_acc}

    def train(self) -> list[float]:
        epoch_times = []
        start = time.time()
        from tqdm.auto import trange
        if self._is_rank_0:
            log.info(', '.join([
                f'self.device: {self.device}',
                f'self.dtype: {self.dtype}',
            ]))
        for epoch in trange(
                1,
                self.config.trainer.epochs + 1,
                disable=(not self._is_rank_0),
                dynamic_ncols=True,
                leave=True,
                desc='Training',
        ):
            t0 = time.time()
            metrics = self.train_epoch(epoch)
            epoch_times.append(time.time() - t0)

            if epoch % self.config.trainer.logfreq and self._is_rank_0:
                acc = self.test()
                astr = f'[TEST] Accuracy: {100.0 * acc:.0f}%'
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)
                summary = '  '.join([
                    '[TRAIN]',
                    f'epoch {epoch} took: {epoch_times[-1]:.4f}s',
                    f'loss={metrics["loss"]:.8f}',
                    f'acc={metrics["acc"] * 100.0:.0f}%'
                ])
                if self.wbrun is not None:
                    self.wbrun.log(
                        {
                            'epoch/epoch': epoch,
                            **{f'epoch/{k}': v for k, v in metrics.items()}
                        }
                    )
                log.info((sep := '-' * len(summary)))
                log.info(summary)
                log.info(sep)
            if self.wbrun is not None:
                self.wbrun.log({'time_per_epoch': (time.time() - t0)})

        rstr = f'[{self.rank}] ::'
        if self.rank == 0:
            log.info(' '.join([
                rstr,
                f'Total training time: {time.time() - start} seconds'
            ]))
            avg_over = min(5, (len(epoch_times) - 1))
            avg_epoch_time = np.mean(epoch_times[-avg_over:])
            log.info(' '.join([
                rstr,
                'Average time per epoch in the last'
                f' {avg_over}: {avg_epoch_time}'
            ]))
        return epoch_times

    def test(self) -> float:
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.data['test']['loader']:
                if torch.cuda.is_available():
                    data = data.to(self.dtype).cuda()
                    target = target.to(self.dtype).cuda()

                probs = self.model(data)
                probs = probs.to(self.device).to(self.dtype)
                _, predicted = probs.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total
        if self.rank == 0:
            try:
                self.wbrun.log({'test/acc': correct / total})  # type:ignore
            except Exception:
                pass

        return acc
