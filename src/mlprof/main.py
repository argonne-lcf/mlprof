"""
main.py

Main entry point for `mlprof`
"""
from __future__ import absolute_import, annotations, division, print_function
import sys

import hydra
from omegaconf import DictConfig
import wandb

from mlprof import get_logger
import mlprof.utils.dist as dist
from mlprof.utils.dist import cleanup, setup_torch
log = get_logger(__name__)


def train_mnist(cfg: DictConfig) -> float:
    from mlprof.trainers.pytorch.trainer import Trainer
    trainer = Trainer(config=cfg)
    _ = trainer.train()
    return trainer.test()


def run(cfg: DictConfig) -> float:
    rank = setup_torch(
        seed=cfg.get('seed', 12345),
        precision=cfg.get('precision', 'fp32'),
        backend=cfg.backend,
        port=cfg.get('port', '2345')
    )
    if rank == 0:
        if cfg.get('wandb', None) is not None:
            from mlprof.common import setup_wandb
            setup_wandb(cfg)
    return train_mnist(cfg)


# @record
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    run(cfg)
    if wandb.run is not None:
        wandb.run.save('.', policy='end')
    if (backend := str(cfg.get('backend', '')).lower()) == 'ddp':
        cleanup()
    if backend in {'deepspeed', 'ds'}:
        import deepspeed.comm as dscomm
        dscomm.log_summary()


if __name__ == '__main__':
    main()
    sys.exit(0)
