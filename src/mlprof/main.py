"""
main.py

Conains simple implementation illustrating how to use PyTorch DDP for
distributed data parallel training.
"""
from __future__ import absolute_import, annotations, division, print_function

import hydra
# from hydra.utils import instantiate
# import numpy as np
from omegaconf import DictConfig
import wandb

from mlprof.utils.pylogger import get_pylogger
from mlprof.utils.dist import cleanup, setup_torch


log = get_pylogger(__name__)


def train_mnist(cfg: DictConfig) -> float:
    from mlprof.trainers.pytorch.trainer import Trainer
    trainer = Trainer(config=cfg)
    _ = trainer.train()
    test_acc = trainer.test()
    return test_acc


def run(cfg: DictConfig) -> float:
    _ = setup_torch(
        seed=cfg.get('seed', 12345),
        precision=cfg.get('precision', 'fp32'),
        backend=cfg.backend,
        port=cfg.get('port', '2345')
    )

    test_acc = train_mnist(cfg)
    return test_acc


# @record
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    run(cfg)
    if str(cfg.get('backend', '')).lower() == 'ddp':
        cleanup()


if __name__ == '__main__':
    wandb.require('service')
    main()
