"""
configs.py

Implements various configuration objects.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict, dataclass, field
import datetime
import logging
from abc import abstractmethod
import os
import json
from pathlib import Path
import random
from typing import Any, Optional, Sequence
import rich.repr

# from ConfigSpace.api.distributions import Distribution
from hydra.core.config_store import ConfigStore
import numpy as np
# import pytorch_lightning as pl
import torch
# import torch.utils.data as data


log = logging.getLogger(__name__)


HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONF_DIR = HERE.joinpath('conf')
LOGS_DIR = HERE.joinpath('logs')
AIM_DIR = HERE.joinpath('.aim')
OUTPUTS_DIR = HERE.joinpath('outputs')
DATA_DIR = HERE.joinpath('data')

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


MODELS = {}

SYNONYMS = {
    'p': 'pytorch',
    'pt': 'pytorch',
    'torch': 'pytorch',
    'pytorch': 'pytorch',
    # -------------------------
    't': 'tensorflow',
    'tf': 'tensorflow',
    'tflow': 'tensorflow',
    'tensorflow': 'tensorflow',
    # -------------------------
    'h': 'horovod',
    'hv': 'horovod',
    'hvd': 'horovod',
    'horovod': 'horovod',
    # -------------------------
    'DDP': 'DDP',
    'ddp': 'DDP',
    # -------------------------
    'ds': 'deepspeed',
    'deepspeed': 'deepspeed',
}


# def add_to_outdirs_file(outdir: os.PathLike):
#     with open(OUTDIRS_FILE, 'a') as f:
#         f.write(Path(outdir).resolve().as_posix())
#         f.write('\n')


def seed_everything(seed: int):
    # pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type:ignore
        torch.backends.cudnn.benchmark = False     # type:ignore
        torch.use_deterministic_algorithms(True)


def get_timestamp(fstr: Optional[str] = None):
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def list_to_str(x: Sequence[int | float | Any]) -> str:
    if isinstance(x[0], int):
        return '-'.join([str(int(i)) for i in x])
    elif isinstance(x[0], float):
        return '-'.join([f'{i:2.1f}' for i in x])
    else:
        return '-'.join([str(i) for i in x])


def add_to_outdirs_file(outdir: os.PathLike):
    with open(OUTDIRS_FILE, 'a') as f:
        f.write(Path(outdir).resolve().as_posix() + '\n')


@dataclass
@rich.repr.auto
class BaseConfig:
    def update(self, params: dict[str, Any]):
        for key, val in params.items():
            old_val = getattr(self, key, None)
            if old_val is not None:
                log.info(f'Updating {key} from: {old_val} to {val}')
                setattr(self, key, val)

    @abstractmethod
    def to_str(self) -> str:
        pass

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def get_config(self) -> dict:
        return asdict(self)

    def asdict(self) -> dict:
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def to_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).resolve().open('w') as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).resolve().open('r') as f:
            config = json.load(f)

        self.__init__(**config)


@dataclass
class ConvNetworkConfig(BaseConfig):
    drop1: float
    drop2: float
    filters1: int
    filters2: int
    hidden_size: int
    activation_fn: str = 'relu'

    def to_str(self) -> str:
        return '_'.join([
            f'dp1-{self.drop1:.3f}',
            f'dp2-{self.drop2:.3f}',
            f'f1-{self.filters1}',
            f'f2-{self.filters2}',
            f'n-{self.hidden_size}',
        ])


@dataclass
class Conv2dConfig(BaseConfig):
    filters: int
    kernel_size: int

    def to_str(self) -> str:
        return '_'.join([
            f'f-{self.filters}',
            f'k-{self.kernel_size}'
        ])


@dataclass
class LocalizationConfig:
    conv1: Conv2dConfig
    conv2: Conv2dConfig


@dataclass
class SpatialTransformerConfig(BaseConfig):
    conv1: Conv2dConfig
    conv2: Conv2dConfig
    hidden1: int = 50
    hidden2: int = 10

    dropout: Optional[float] = None


@dataclass
class NetworkConfig(BaseConfig):
    pass


@dataclass
class DataConfig(BaseConfig):
    batch_size: int = 128
    dataset: str = 'MNIST'

    def __post_init__(self):
        assert self.dataset in ['MNIST', 'FashionMNIST']

    def to_str(self) -> str:
        return f'dset-{self.dataset}'


@dataclass
class TrainerConfig(BaseConfig):
    lr_init: float
    logfreq: int = 10
    epochs: int = 5
    num_threads: int = 16
    # seed: int = 9992
    # batch_size: int
    # dataset: str = 'MNIST'

    def scale_lr(self, factor: int) -> float:
        return self.lr_init * factor

    def to_str(self) -> str:
        return ''


def load_ds_config(fpath: os.PathLike) -> dict:
    ds_config_path = Path(fpath)
    with ds_config_path.open('r') as f:
        ds_config = json.load(f)

    return ds_config


@dataclass
class FlopsProfiler:
    enabled: bool = True
    profile_step: int = 1
    module_depth: int = -1
    top_modules: int = 1
    detailed: bool = True
    output_file: Optional[os.PathLike] = None


@dataclass
class OptimizerParams:
    lr: float = 0.001
    betas: list[float] = field(default_factory=list)
    eps: float = 1e-8
    weight_decay: float = 3e-7

    def __post_init__(self):
        if self.betas == []:
            self.betas = [0.8, 0.999]
            log.warning('Betas not set, using defaults: [0.8, 0.999]')


@dataclass
class OptimizerConfig:
    type: str = 'AdamW'
    params: Optional[dict | OptimizerParams] = None

    def __post_init__(self):
        if isinstance(self.params, dict):
            self.params = OptimizerParams(**self.params)


@dataclass
class SchedulerConfig:
    type: str = "WarmupLR"


@dataclass
class DeepSpeedConfig:
    dump_state: bool = True
    prescale_gradients: bool = False
    wall_clock_breakdown: bool = True


@dataclass
class ExperimentConfig(BaseConfig):
    data: DataConfig
    trainer: TrainerConfig
    network: ConvNetworkConfig
    wandb: Any
    framework: str = 'pytorch'
    backend: str = 'DDP'
    seed: int = 1234
    precision: str = 'float32'
    autocast: bool = True
    ds_config_path: Optional[Any] = None
    compression: str = 'fp16'
    # compression: Optional[str] = None
    # ds_config: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.backend = SYNONYMS[self.backend]
        self.framework = SYNONYMS[self.framework]
        if self.ds_config_path is None:
            self.ds_config_path = Path(
                (CONF_DIR).joinpath('ds_config.json')
            )
        # self.ds_config = {}
        # if (
        #         self.ds_config is None and
        #         self.backend.lower() in ['ds', 'deepspeed']
        # ):
        #     if self.backend.lower() in ['ds', 'deepspeed']:
        #         if self.ds_config_path is None:
        #             self.ds_config_path = CONF_DIR.joinpath(
        #                 'ds_config.json'
        #             )
        #             log.warning(
        #                 f'Using DeepSpeed config from: '
        #                 f'{self.ds_config_path}'
        #             )
        #         # assert self.ds_config_path is not None
        #         self.ds_config_path = Path(self.ds_config_path)
        #         self.ds_config = self.load_ds_config(
        #             self.ds_config_path
        #         )
        #     # if self.ds_config_path is not None:
        #     #     fpath = Path(self.ds_config_path)
        #     #     if fpath.is_file():
        #     #         log.warning(f'Loading DeepSpeed config from: {fpath}')
        #     #         self.ds_config = load_ds_config(fpath)

    def to_str(self) -> str:
        return '_'.join([
            f'{self.framework}',
            # f'{}
            self.network.to_str(),
        ])

    def load_ds_config(self, fpath: Optional[os.PathLike]) -> dict:
        if self.backend.lower() not in ['ds', 'deepspeed']:
            return {}
        fname = self.ds_config_path if fpath is None else fpath
        assert fname is not None
        cpath = Path(fname)
        if cpath.exists():
            with cpath.open('r') as f:
                ds_config = json.load(f)
            return ds_config
        return {}


def get_config(overrides: Optional[list[str]] = None):
    from hydra import (
        initialize_config_dir,
        compose
    )
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    overrides = [] if overrides is None else overrides
    with initialize_config_dir(
            CONF_DIR.absolute().as_posix(),
            # version_base=None,
    ):
        cfg = compose(
            'config',
            overrides=overrides,
            # return_hydra_config=True,
        )

    return cfg


cs = ConfigStore.instance()
cs.store(
    name='experiment_config',
    node=ExperimentConfig
)
