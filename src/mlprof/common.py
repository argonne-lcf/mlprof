"""
common.py
"""
from __future__ import absolute_import, annotations, division, print_function
import os


from mlprof.configs import PROJECT_DIR
from omegaconf import DictConfig, OmegaConf


def setup_wandb(
    cfg: DictConfig
) -> dict:
    import torch
    wbrun = None
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
        # ngpus_env = os.environ.get('NGPUS', SIZE)
        # if ngpus_env != SIZE:
        #     log.warning('$NGPUS != SIZE')
        #     log.warning(f'NRANKS: {ngpus_env}')
        #     log.warning(f'SIZE: {SIZE}')
        # wbrun.config['NRANKS'] = SIZE  # os.environ.get('NRANKS', SIZE)
        # wbrun.config['hostname'] = MASTER_ADDR
        wbrun.config['device'] = (
            'gpu' if torch.cuda.is_available() else 'cpu'
        )

    return {'run': wbrun}
