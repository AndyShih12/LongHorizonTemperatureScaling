# IMPORTS
import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from runner_gpt2 import Runner
import torch

# A logger for this file
import logging
LOG = logging.getLogger(__name__)
import wandb

@hydra.main(config_path="conf", config_name="config_gpt2")
def main(cfg: DictConfig):
    rank = int(os.environ["LOCAL_RANK"])
    nranks = int(os.environ["WORLD_SIZE"])

    cfg.local_rank = rank
    cfg.world_size = nranks

    # relative to hydra path
    os.makedirs(cfg.model_dir, exist_ok=True)

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"env://",
    )

    if cfg.local_rank == 0:
        config_dict = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(project="lhts", entity="", config=config_dict)

    LOG.info(os.getcwd())
    LOG.info(OmegaConf.to_yaml(cfg))

    runner = Runner(cfg)

    if cfg.mode == 'train':
        runner.train()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()