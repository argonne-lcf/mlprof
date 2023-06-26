"""
common.py
"""
from __future__ import absolute_import, annotations, division, print_function
import os

from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.util import generate_id

from mlprof.configs import PROJECT_DIR
from mlprof.configs import ExperimentConfig
import mlprof.utils.dist as dist


ENV_FILTERS = [
    'PS1',
    'LSCOLORS',
    'LS_COLORS',
]

ENV_PREFIXES = [
    '_ModuleTable',
    'BASH_FUNC_',
]


def start_wandb_run(cfg: DictConfig | ExperimentConfig):
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
    # wbrun.log_code(cfg.get('work_dir', PROJECT_DIR))
    wbrun.log_code(PROJECT_DIR)
    wbrun.config.update(OmegaConf.to_container(cfg, resolve=True))
    wbrun.config['run_id'] = run_id
    wbrun.config['logdir'] = os.getcwd()
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


def setup_wandb(cfg: DictConfig | ExperimentConfig):
    # wbrun = None
    if isinstance(cfg, DictConfig):
        wbcfg = cfg.get('wandb', None)
    elif isinstance(cfg, ExperimentConfig):
        wbcfg = cfg.wandb
    else:
        raise TypeError(f'type: {type(cfg)}')
    if wbcfg is not None:
        start_wandb_run(cfg=cfg)
        # ngpus_env = os.environ.get('NGPUS', SIZE)
        # if ngpus_env != SIZE:
        #     log.warning('$NGPUS != SIZE')
        #     log.warning(f'NRANKS: {ngpus_env}')
        #     log.warning(f'SIZE: {SIZE}')
        # wbrun.config['NRANKS'] = SIZE  # os.environ.get('NRANKS', SIZE)
        # wbrun.config['hostname'] = MASTER_ADDR
        # import torch
        # wbrun.config['device'] = (
        #     'gpu' if torch.cuda.is_available() else 'cpu'
        # )

    # return {'run': wbrun}
    # return wbrun
