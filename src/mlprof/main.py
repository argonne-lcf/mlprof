"""
main.py

Conains simple implementation illustrating how to use PyTorch DDP for
distributed data parallel training.
"""
from __future__ import absolute_import, annotations, division, print_function
import os
import socket
import time
from typing import Any, Optional

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.nn as nn
# from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.utils.data.distributed
import wandb

from mlprof.configs import PROJECT_DIR
from mlprof.configs import NetworkConfig
from mlprof.network.pytorch.network import Net
from mlprof.utils.pylogger import get_pylogger
from mlprof.utils.dist import cleanup, init_process_group, setup_torch


log = get_pylogger(__name__)


try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = int(              # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        os.environ.get(            # ┃ Should be set by torch,  ┃
            'LOCAL_RANK',          # ┃ otherwise we should:     ┃
            os.environ.get(        # ┃━━━━━━━━━━━━━━━━━━━━━━━━━━┃
                'PMI_LOCAL_RANK',  # ┃   1. check for Polaris   ┃
                os.environ.get(    # ┃   2. check for ThetaGPU  ┃
                    'OMPI_COMM_WORLD_LOCAL_RANK',  # ━━━━━━━━━━━┛
                    '0'
                )
            )
        )
    )
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)


def metric_average(val: torch.Tensor):
    if (WITH_DDP):
        # Sum everything and divide by the total size
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE

    return val


def setup_wandb(
    cfg: DictConfig
) -> dict:
    wbrun = None
    # if RANK == 0 and LOCAL_RANK == 0:
    wbcfg = cfg.get('wandb', None)
    if wbcfg is not None:
        import wandb
        from wandb.util import generate_id
        run_id = generate_id()
        wbrun = wandb.init(
            dir=os.getcwd(),
            id=run_id,
            mode='online',
            resume='allow',
            save_code=True,
            project=cfg.wandb.setup.project,
        )
        assert wbrun is not None and wbrun is wandb.run
        wbrun.log_code(cfg.get('work_dir', PROJECT_DIR))
        wbrun.config.update(OmegaConf.to_container(cfg, resolve=True))
        wbrun.config['run_id'] = run_id
        wbrun.config['logdir'] = os.getcwd()
        ngpus_env = os.environ.get('NGPUS', SIZE)
        # if ngpus_env != SIZE:
        #     log.warning('$NGPUS != SIZE')
        #     log.warning(f'NRANKS: {ngpus_env}')
        #     log.warning(f'SIZE: {SIZE}')
        wbrun.config['NRANKS'] = SIZE  # os.environ.get('NRANKS', SIZE)
        wbrun.config['hostname'] = MASTER_ADDR
        wbrun.config['device'] = (
            'gpu' if torch.cuda.is_available() else 'cpu'
        )

    return {'run': wbrun}


def train_mnist(cfg: DictConfig, wbrun: Optional[Any] = None) -> float:
    from mlprof.trainers.pytorch.trainer import Trainer
    from mlprof.configs import ExperimentConfig
    start = time.time()
    config: ExperimentConfig = instantiate(cfg)
    assert isinstance(config, ExperimentConfig)
    # config.update_dist_config(
    #     size=SIZE,
    #     rank=RANK,
    #     local_rank=LOCAL_RANK
    # )
    # tconfig = instantiate(cfg.get('trainer'))
    # net_config = instantiate(cfg.get('network'))
    # assert isinstance(tconfig, TrainerConfig)
    # assert isinstance(net_config, NetworkConfig)
    # xshape = (tconfig.batch_size, *(1, *[28, 28]))
    # model = build_model(net_config, xshape)
    trainer = Trainer(config=config, wbrun=wbrun)
    epoch_times = []
    for epoch in range(1, config.trainer.epochs + 1):
        t0 = time.time()
        metrics = trainer.train_epoch(epoch)
        epoch_times.append(time.time() - t0)

        if epoch % config.trainer.logfreq and RANK == 0:
            acc = trainer.test()
            astr = f'[TEST] Accuracy: {100.0 * acc:.0f}%'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={metrics["loss"]:.4f}',
                f'acc={metrics["acc"] * 100.0:.0f}%'
            ])
            if wbrun is not None:
                wbrun.log(
                    {
                        'epoch/epoch': epoch,
                        **{f'epoch/{k}': v for k, v in metrics.items()}
                    }
                )
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)
        if wbrun is not None:
            wbrun.log({'time_per_epoch': (time.time() - t0)})

    rstr = f'[{RANK}] ::'
    if RANK == 0:
        log.info(' '.join([
            rstr,
            f'Total training time: {time.time() - start} seconds'
        ]))
        avg_over = min(5, (len(epoch_times) - 1))
        avg_epoch_time = np.mean(epoch_times[-avg_over:])
        log.info(' '.join([
            rstr,
            f'Average time per epoch in the last {avg_over}: {avg_epoch_time}'
        ]))

    test_acc = trainer.test()
    return test_acc


def run(cfg: DictConfig) -> float:
    # from mlprof.common import setup_wandb
    wb = {'run': None}
    if RANK == 0 and LOCAL_RANK == 0:
        wb = setup_wandb(cfg)
    # backend = 'NCCL' if torch.cuda.is_available() else 'gloo'
    # init_process_group(RANK, world_size=SIZE, backend=backend)
    _ = setup_torch(
        seed=cfg.get('seed', 12345),
        precision=cfg.get('precision', 'fp32'),
        backend=cfg.backend,
        port=cfg.get('port', '2345')
    )
    test_acc = train_mnist(cfg, wb['run'])
    return test_acc


# @record
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    run(cfg)
    cleanup()


if __name__ == '__main__':
    wandb.require('service')
    main()
