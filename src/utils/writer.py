import glob
import shutil
from pathlib import Path

import numpy as np
import torch
from matplotlib.pyplot import Figure

import wandb
from src.utils.misc import get_image


class WandBWriter:
    def __init__(
            self,
            run_cfg: dict,
            project: str,
            group: str = None,
            run_name: str = None,
            run_id: str = None,
            mode: str = "online"
        ):
        wandb.login()
        wandb.init(project=project, group=group, config=run_cfg, name=run_name, id=run_id, mode=mode)

        self._step = 0
        self._section = None

    def _name(self, name: str):
        return (f"{self._section}/{name}" if self._section is not None else name)

    def set_step(self, step: int, section: str="train"):
        self._step = step
        self._section = section

    def add_scalar(self, scalar_name: str, scalar: int | float):
        wandb.log({self._name(scalar_name): scalar}, step=self._step)

    def add_scalars(self, scalars: dict):
        wandb.log({self._name(scalar_name): scalar for scalar_name, scalar in scalars.items()}, step=self._step)

    def add_audio(self, audio_name: str, audio: torch.Tensor, fs: int = None, normalise: bool = True):
        audio = audio.detach().cpu().numpy()
        if normalise:
            audio = audio / np.max(np.abs(audio))
        wandb.log({self._name(audio_name): wandb.Audio(audio, sample_rate=fs)}, step=self._step)

    def add_fig(self, fig_name: str, fig: Figure):
        wandb.log({self._name(fig_name): wandb.Image(get_image(fig))}, step=self._step)

    def add_figs(self, figs: dict):
        wandb.log({self._name(fig_name): wandb.Image(get_image(fig)) for fig_name, fig in figs.items()}, step=self._step)

    def add_histogram(self, histogram_name: str, data: torch.Tensor, num_bins: int = 100):
        data = data.detach().cpu().numpy()
        wandb.log({self._name(histogram_name): wandb.Histogram(data, num_bins=num_bins)}, step=self._step)

    def add_checkpoints(self, save_dir: Path):
        checkpoint_names = glob.glob("checkpoint_*.pt", root_dir=save_dir)
        for checkpoint_name in checkpoint_names:
            src_file = save_dir / checkpoint_name
            dst_file = Path(wandb.run.dir) / checkpoint_name
            shutil.copy(src_file, dst_file)
            wandb.save(dst_file, wandb.run.dir, policy="end")
