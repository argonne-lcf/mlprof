"""
trainers/trainer.py

Implements BaseTrainer class
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from mlprof.configs import ExperimentConfig
# from mlprof.utils.pylogger import get_pylogger

# log = get_pylogger(__name__)
from mlprof import get_logger
log = get_logger(__name__)


class BaseTrainer(ABC):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig | dict,
    ) -> None:
        self.config: ExperimentConfig = (
            cfg if isinstance(cfg, ExperimentConfig)
            else instantiate(cfg)
        )

    @abstractmethod
    def train(self):
        pass
