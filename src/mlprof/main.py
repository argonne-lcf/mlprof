"""
main.py

Conains simple implementation illustrating how to use PyTorch DDP for
distributed data parallel training.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Any, Optional

import hydra
# from hydra.utils import instantiate
# import numpy as np
from omegaconf import DictConfig
import wandb

from mlprof.utils.pylogger import get_pylogger
from mlprof.utils.dist import cleanup, setup_torch


log = get_pylogger(__name__)


def train_mnist(cfg: DictConfig, wbrun: Optional[Any] = None) -> float:
    from mlprof.trainers.pytorch.trainer import Trainer
    # from mlprof.configs import ExperimentConfig
    # config: ExperimentConfig = instantiate(cfg)
    # assert isinstance(config, ExperimentConfig)
    trainer = Trainer(config=cfg, wbrun=wbrun)
    _ = trainer.train()
    test_acc = trainer.test()
    return test_acc


def run(cfg: DictConfig) -> float:
    from mlprof.common import setup_wandb
    dsetup = setup_torch(
        seed=cfg.get('seed', 12345),
        precision=cfg.get('precision', 'fp32'),
        backend=cfg.backend,
        port=cfg.get('port', '2345')
    )
    wb = {'run': None}
    if dsetup['rank'] == 0 and dsetup['local_rank'] == 0:
        wb = setup_wandb(cfg)

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
