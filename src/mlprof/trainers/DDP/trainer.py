"""
ddp/trainer.py
"""
from __future__ import absolute_import, annotations, division, print_function
import time
import os
from typing import Optional, Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import print_json
from pathlib import Path
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

from mlprof.trainers.trainer import BaseTrainer
from mlprof.configs import CONF_DIR, DATA_DIR, ExperimentConfig, NetworkConfig
from mlprof.network.pytorch.network import Net
from mlprof.utils.pylogger import get_pylogger
from mlprof.utils.dist import setup_torch_distributed

log = get_pylogger(__name__)

Optimizer = torch.optim.Optimizer


def metric_average(val: torch.Tensor, size: int = 1):
    try:
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
    except Exception as exc:
        log.exception(exc)
        pass

    return val / size


def load_ds_config(fpath: os.PathLike) -> dict:
    ds_config_path = Path(fpath)
    with ds_config_path.open('r') as f:
        ds_config = json.load(f)

    return ds_config


class Trainer:
    def __init__(
            self,
            config: ExperimentConfig | dict | DictConfig,
            wbrun: Optional[Any] = None,
            scaler: Optional[GradScaler] = None,
            model: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.config: ExperimentConfig = (
            config if isinstance(config, ExperimentConfig)
            else instantiate(config)
        )
        # else:
        #     raise TypeError(
        #         'Expected `config` to be of type: '
        #         '`dict | DictConfig | ExperimentConfig`'
        #     )

        # self.rank = dist.get_rank()
        # self.world_size = dist.get_world_size()
        # self.local_size = 1
        # self._ngpus = 0
        dsetup = setup_torch_distributed(self.config.backend)
        self._dtype = torch.get_default_dtype()
        self._device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.world_size = dsetup['size']
        self.rank = dsetup['rank']
        self.local_rank = dsetup['local_rank']
        self._is_chief = (self.local_rank == 0 and self.rank == 0)
        if torch.cuda.is_available():
            self.local_size = torch.cuda.device_count()
            self._ngpus = self.world_size * torch.cuda.device_count()

        if scaler is None:
            self.scaler = None

        assert isinstance(self.config, ExperimentConfig)
        self._global_step = 0

        self.loss_fn = nn.CrossEntropyLoss()

        # self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # self.rank = 0
        # self._ngpus = 1
        # self.world_size = 1
        # self.setup_torch()
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

        if torch.cuda.is_available():
            self.model.cuda()

        self.wbrun = wbrun
        if self.wbrun is not None:
            log.warning(f'Caught wandb.run from: {self.rank}')
            self.wbrun.watch(
                self.model,
                log='all',
                criterion=self.loss_fn,
                log_graph=True,
                log_freq=self.config.trainer.logfreq,
            )

        self.model_engine = None
        self.use_fp16 = False
        self.ds_config = None
        if self.config.backend == 'DDP':
            self.model_engine = DDP(self.model)
            self.optimizer = self._optimizer

        elif self.config.backend.lower() in ['ds', 'deepspeed']:
            import deepspeed
            ds_config_path = Path(CONF_DIR).joinpath('ds_config.json')
            self.ds_config = load_ds_config(ds_config_path)
            log.info(f'Loaded DeepSpeed config from: {ds_config_path}')
            if self._is_chief:
                print_json(json.dumps(self.ds_config, indent=4))

            if self.ds_config['fp16']['enabled']:
                self.model = self.model.to(torch.half)

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
            self.optimizer = optimizer
            self.use_fp16 = self.model_engine.fp16_enabled()
            if self.use_fp16:
                self._dtype = torch.half

        elif self.config.backend in ['hvd', 'horovod']:
            import horovod.torch as hvd
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

        else:
            self.model_engine = None
            self.optimizer = self._optimizer
            raise ValueError(
                f'Unexpected value for `config.backend`: {self.config.backend}'
            )

    def build_model(self, net_config: NetworkConfig) -> nn.Module:
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
        torch.manual_seed(self.config.trainer.seed)
        # if self.device == 'gpu':
        if torch.cuda.is_available():
            # DDP: pin GPU to local rank
            # torch.cuda.set_device(int(LOCAL_RANK))
            torch.cuda.manual_seed(self.config.trainer.seed)

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
        return {
            'train': train_dataset,
            'test': test_dataset,
        }

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

        if self._device == 'cuda':
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
            data, target = data.cuda(), target.cuda()

        assert self.optimizer is not None
        self.optimizer.zero_grad()
        probs = self.model(data)
        loss = self.loss_fn(probs, target)

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

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

    def train_epoch(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        # start = time.time()
        running_acc = torch.tensor(0.)
        running_loss = torch.tensor(0.)
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
        training_acc = metric_average(running_acc, size=self._ngpus)
        loss_avg = metric_average(running_loss, size=self._ngpus)
        if self.rank == 0:
            assert (
                self.wbrun is not None
                and self.wbrun is wandb.run  # type:ignore
            )
            self.wbrun.log({'train/loss': loss_avg, 'train/acc': training_acc})

        return {'loss': loss_avg, 'acc': training_acc}

    def test(self) -> float:
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.data['test']['loader']:
                if self._device == 'cuda':
                    data, target = data.cuda(), target.cuda()

                probs = self.model(data)
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
