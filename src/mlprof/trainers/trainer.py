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
from mlprof.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BaseTrainer(ABC):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig | dict,
            # keep: Optional[str | list[str]] = None,
            # skip: Optional[str | list[str]] = None
            # config: ExperimentConfig | DictConfig | dict,
            # wbrun: Optional[Any] = None,
            # scaler: Optional[GradScaler] = None,
            # model: Optional[torch.nn.Module] = None,
            # optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        self.config: ExperimentConfig = (
            cfg if isinstance(cfg, ExperimentConfig)
            else instantiate(cfg)
        )
        # if isinstance(config, (dict, DictConfig)):
        #     config = instantiate(config)

        # assert isinstance(config, ExperimentConfig)
        # self.config = config
        # self.scaler = scaler
        # self._global_step = 0
        # self.loss_fn = nn.CrossEntropyLoss()
        # self._device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # self._rank = 0
        # self._ngpus = 1
        # self._world_size = 1
        # self.data = self.setup_data()
        # if model is None:
        #     self.model = self.build_model(self.config.network)

    # @abstractmethod
    # def eval_step(self, input: Any):
    #     pass

    # @abstractmethod
    # def train_step(self, input: Any):
    #     pass

    @abstractmethod
    def train(self):
        pass
