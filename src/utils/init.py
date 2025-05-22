import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv, set_key
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.io import ROOT_PATH
from wandb.util import generate_id


def set_random_seed(seed: int):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    update_env("PYTHON_SEED", str(seed))


def setup_saving(cfg: DictConfig, cfg_dir: Path, cfg_name: str):
    resume = (cfg.save_dir in str(cfg_dir))
    if resume:
        save_dir = cfg_dir
    else:
        run_id = generate_id()
        with open_dict(cfg):
            cfg.writer.run_id = run_id
        datetime_str = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        save_dir = ROOT_PATH / cfg.save_dir / (datetime_str + "_" + cfg_name)
        save_dir.mkdir(exist_ok=True, parents=True)
        log_git(save_dir)
        OmegaConf.save(cfg, save_dir / "cfg.yaml", resolve=True)
    update_env("PYTHON_LAST_SAVE_DIR", str(save_dir))
    return save_dir, resume


def setup_logging(save_dir: Path, resume: bool = False):
    log_file = save_dir / "out.log"
    logging.basicConfig(
        filename=log_file,
        filemode=("w" if resume is False else "a"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        force=True
    )


def log_git(save_dir: Path):
    commit_file = save_dir / "git_commit.txt"
    patch_file = save_dir / "git_diff.patch"
    with commit_file.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_file.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def update_env(key: str, value: str):
    os.environ[key] = value
    dotenv_file = ROOT_PATH / ".env"
    dotenv_file.touch(exist_ok=True)
    set_key(dotenv_file, key, value)


def load_env():
    dotenv_file = ROOT_PATH / ".env"
    if dotenv_file.exists():
        load_dotenv(dotenv_file)