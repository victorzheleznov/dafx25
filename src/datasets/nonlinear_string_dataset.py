import copy
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from src.datasets import BaseDataset
from src.generators import LinearString, NonlinearString
from src.models import ModalSystem
from src.utils.excitation import Excitation
from src.utils.init import log_git
from src.utils.io import ROOT_PATH, write_wav
from src.utils.misc import calc_md5, gen_from_range
from src.utils.writer import WandBWriter

MAX_NUM_PLOTS = 10
NUM_POINTS = 1000
START_MODE = 0
NUM_MODES = 3
NUM_PERIODS = 5


class NonlinearStringDataset(BaseDataset):
    """Class for nonlinear string dataset."""
    def __init__(
            self,
            fs: int,
            dur: float,
            method: str,
            num_modes: int,
            gamma_range: list[float],
            kappa_range: list[float],
            sigma0_range: list[float],
            sigma1_range: list[float],
            xe_range: list[float],
            xo_range: list[float],
            exc_amp_range: list[float],
            exc_dur_range: list[float],
            exc_type: int,
            num_variations: int = 10,
            batch_size: int = None,
            device: str = "cpu"
        ) -> None:
        """Parameters
        ----------
        fs : int
            Sampling rate [Hz].
        dur : float
            Simulation duration [sec].
        method : str
            Simulation method (e.g., "verlet").
        num_modes : int
            Number of modes.
        gamma_range : list[float]
            Range for randomised wavespeeds.
        kappa_range : list[float]
            Range for randomised stiffness parameters.
        sigma0_range : list[float]
            Range for randomised frequency-independent damping parameters.
        sigma1_range : list[float]
            Range for randomised frequency-dependent damping parameters.
        xe_range : list[float]
            Range for randomised excitation positions.
        xo_range : list[float]
            Range for randomised output positions.
        exc_amp_range : list[float]
            Range for randomised excitation amplitudes.
        exc_dur_range : list[float]
            Range for randomised excitation durations.
        exc_type : int
            Excitation type: 1 - pluck, 2 - strike.
        num_variations : int
            Number of variations (i.e., number of different strings) to generate.
        batch_size : int
            Batch size for dataset generation (if None then equal to number of variations).
        device : str
            Device (e.g., "cpu" or "cuda") used to generate data.

        Notes
        -----
        The dataset is generated upon initialisation of this class. It is saved to a folder with an unique name based on
        specified generation parameters. If the path exists, it is assumed that the data is already generated and
        the dataset is loaded without generation.
        """
        super().__init__()

        # parse parameters
        self._gamma_range = gamma_range
        self._kappa_range = kappa_range
        self._sigma0_range = sigma0_range
        self._sigma1_range = sigma1_range
        self._xe_range = xe_range
        self._xo_range = xo_range
        self._exc_amp_range = exc_amp_range
        self._exc_dur_range = exc_dur_range
        self._exc_st = 0.0
        self._exc_type = exc_type
        self._num_variations = num_variations
        self._batch_size = min(batch_size, num_variations) if batch_size else num_variations
        self._device = torch.device(device)

        # create generators
        self._gen = NonlinearString(
            fs,
            dur,
            method,
            num_modes
        ).to(torch.double).to(self._device)

        self._gen_lin = LinearString(
            fs,
            dur,
            method,
            num_modes
        ).to(torch.double).to(self._device)

        # specify data paths
        self._dataset_dir = (
            ROOT_PATH
            / "data"
            / "nonlinear_string"
            / str(self)
        )
        self._audio_dir = self._dataset_dir / "audio"
        self._data_dir = self._dataset_dir / "data"
        self._meta_file = self._dataset_dir / "meta.pt"

        # synthesise or load dataset
        if not self._dataset_dir.exists():
            self._dataset_dir.mkdir(parents=True)
            self._audio_dir.mkdir(parents=True)
            self._data_dir.mkdir(parents=True)
            self._synthesise_dataset()
            self._save_dataset()
            log_git(self._dataset_dir)
            print(f"Dataset saved to {self._dataset_dir}")
        else:
            self._load_dataset()
            print(f"Loaded dataset from {self._dataset_dir}")

    def __str__(self):
        return f"{self._gen.method}_{self._gen.fs:d}Hz_{self._gen.dur:g}sec_{self._gen.num_modes:d}modes_{self.md5}"

    def __len__(self):
        return self._num_variations

    def __getitem__(self, idx):
        # get metadata
        instance = self._meta[idx]

        # get nonlinear string output
        data = torch.load(self._data_dir / f"{idx}.pt", weights_only=True)
        for key in data.keys():
            instance["target_" + key] = data[key]

        instance["idx"] = idx
        return instance

    def _getlin(self, idx):
        # get linear string output
        instance = dict()
        data_lin = torch.load(self._data_dir / f"{idx}_lin.pt", weights_only=True)
        for key in data_lin.keys():
            instance["linear_" + key] = data_lin[key]
        return instance

    def _synthesise_dataset(self):
        # generate initial conditions
        y0 = torch.zeros((self._num_variations, int(2 * self._gen.num_modes)), dtype=torch.double, device=self._device)

        # generate physical parameters
        gamma = gen_from_range(self._gamma_range, (self._num_variations,), dtype=torch.double, device=self._device)
        kappa = gen_from_range(self._kappa_range, (self._num_variations,), dtype=torch.double, device=self._device)
        sigma0 = gen_from_range(self._sigma0_range, (self._num_variations,), dtype=torch.double, device=self._device)
        sigma1 = gen_from_range(self._sigma1_range, (self._num_variations,), dtype=torch.double, device=self._device)
        xe = gen_from_range(self._xe_range, (self._num_variations,), dtype=torch.double, device=self._device)
        xo = gen_from_range(self._xo_range, (self._num_variations,), dtype=torch.double, device=self._device)

        # generate excitation
        exc_amp = gen_from_range(self._exc_amp_range, (self._num_variations,), dtype=torch.double, device=self._device)
        exc_dur = gen_from_range(self._exc_dur_range, (self._num_variations,), dtype=torch.double, device=self._device)
        exc_st = self._exc_st * torch.ones((self._num_variations,), dtype=torch.double, device=self._device)
        exc_type = self._exc_type * torch.ones((self._num_variations,), dtype=torch.int, device=self._device)

        # batched loop
        omega = torch.zeros((self._num_variations, self._gen.num_modes), dtype=torch.double, device=self._device)
        sigma = torch.zeros((self._num_variations, self._gen.num_modes), dtype=torch.double, device=self._device)
        indices = torch.arange(
            start=0,
            end=((self._num_variations // self._batch_size) * self._batch_size)
        ).reshape((-1, self._batch_size))
        indices = [r for r in indices]
        if (self._num_variations % self._batch_size) != 0:
            indices.append(
                torch.arange(
                    start=((self._num_variations // self._batch_size) * self._batch_size),
                    end=self._num_variations
                )
            )
        for r in tqdm(indices):
            # generate nonlinear string output
            data = self._gen(
                y0[r, ...],
                gamma[r, ...],
                kappa[r, ...],
                sigma0[r, ...],
                sigma1[r, ...],
                xe[r, ...],
                xo[r, ...],
                exc_amp=exc_amp[r, ...],
                exc_dur=exc_dur[r, ...],
                exc_st=exc_st[r, ...],
                exc_type=exc_type[r, ...]
            )

            # store modal parameters
            omega[r, ...] = self._gen.omega
            sigma[r, ...] = self._gen.sigma

            # generate linear string output
            data_lin = self._gen_lin(
                y0[r, ...],
                gamma[r, ...],
                kappa[r, ...],
                sigma0[r, ...],
                sigma1[r, ...],
                xe[r, ...],
                xo[r, ...],
                exc_amp=exc_amp[r, ...],
                exc_dur=exc_dur[r, ...],
                exc_st=exc_st[r, ...],
                exc_type=exc_type[r, ...]
            )

            # save batch
            self._save_batch(data, data_lin, r)

        # store metadata
        self._meta = [{
            "fs": self._gen.fs,
            "dur": self._gen.dur,
            "method": self._gen.method,
            "num_modes": self._gen.num_modes,
            "gamma": gamma[idx].item(),
            "kappa": kappa[idx].item(),
            "sigma0": sigma0[idx].item(),
            "sigma1": sigma1[idx].item(),
            "xe": xe[idx].item(),
            "xo": xo[idx].item(),
            "exc_amp": exc_amp[idx].item(),
            "exc_dur": exc_dur[idx].item(),
            "exc_st": exc_st[idx].item(),
            "exc_type": exc_type[idx].item(),
            "y0": y0[idx, :].detach().cpu(),
            "omega": omega[idx, :].detach().cpu(),
            "sigma": sigma[idx, :].detach().cpu()
        } for idx in range(self._num_variations)]

    def _save_batch(self, data: dict[str, torch.Tensor], data_lin: dict[str, torch.Tensor], r: torch.Tensor):
        for i in range(len(r)):
            d = {key: value[i, ...].detach().cpu() for key, value in data.items()}
            d_lin = {key: value[i, ...].detach().cpu() for key, value in data_lin.items()}
            idx = r[i]
            torch.save(d, self._data_dir / f"{idx}.pt")
            torch.save(d_lin, self._data_dir / f"{idx}_lin.pt")
            write_wav(self._audio_dir / f"{idx}.wav", d["w"], self._gen.fs, normalise=True)
            write_wav(self._audio_dir / f"{idx}_lin.wav", d_lin["w"], self._gen_lin.fs, normalise=True)

    def _save_dataset(self):
        torch.save(self._meta, self._meta_file)

    def _load_dataset(self):
        self._meta = torch.load(self._meta_file, weights_only=True)

    def save_pred(self, test_dir: Path, output: torch.Tensor, w: torch.Tensor, idx: torch.Tensor, **batch):
        data_dir = test_dir / "data"
        audio_dir = test_dir / "audio"
        data_dir.mkdir(exist_ok=True, parents=True)
        audio_dir.mkdir(exist_ok=True, parents=True)
        batch_size = output.shape[0]
        for i in range(batch_size):
            d = {"output": output[i, ...].detach().cpu(), "w": w[i, ...].detach().cpu()}
            torch.save(d, data_dir / f"{idx[i]}_pred.pt")
            write_wav(audio_dir / f"{idx[i]}_pred.wav", d["w"], self._gen.fs, normalise=True)

    def plot(self):
        fig, axs = plt.subplots(nrows=2, ncols=1, layout="constrained")
        fig.suptitle("Nonlinear string dataset", size="medium")

        axs[0].set_xlabel("Time [sec]")
        axs[0].set_ylabel("Output wave")

        axs[1].set_xlabel("Time [sec]")
        axs[1].set_ylabel("Excitation")

        for idx in range(len(self)):
            instance = copy.deepcopy(self[idx])
            instance.update(self._getlin(idx))
            self._plot_instance(axs, **instance)

    def plot_batch(
            self,
            target_output: torch.Tensor,
            target_w: torch.Tensor,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
            model: nn.Module = None,
            key: str = None,
            idx: torch.Tensor = None,
            **batch
        ):
        batch_size = target_output.shape[0]
        indices = random.sample(range(batch_size), min(MAX_NUM_PLOTS, batch_size))

        for i in indices:
            title = f"Instance {idx[i].item()}"
            if key is not None:
                title = key.title() + " " + title.lower()

            fig, _ = self._plot_output(
                target_output[i, ...],
                target_w[i, ...],
                output[i, ...] if output is not None else None,
                w[i, ...] if w is not None else None,
                linear_output[i, ...] if linear_output is not None else None,
                linear_w[i, ...] if linear_w is not None else None,
            )
            fig.suptitle(title, size="medium")

            fig, _ = self._plot_velocity(
                target_output[i, ...],
                output[i, ...] if output is not None else None,
                linear_output[i, ...] if linear_output is not None else None,
            )
            fig.suptitle(title, size="medium")

        if isinstance(model, ModalSystem):
            if self._gen.num_modes <= 2:
                self._plot_nonlinearity(self._gen.nl, model.nl, target_output)

    def log_batch(
            self,
            writer: WandBWriter,
            y0: torch.Tensor,
            target_output: torch.Tensor,
            target_w: torch.Tensor,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
            model: nn.Module = None,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None,
            omega: torch.Tensor = None,
            sigma: torch.Tensor = None,
            gamma: torch.Tensor = None,
            xe: torch.Tensor = None,
            xo: torch.Tensor = None,
            idx: torch.Tensor = None,
            **batch
        ):
        title = f"Instance {idx[0].item()}"

        fig, _ = self._plot_output(
            target_output[0, ...],
            target_w[0, ...],
            output[0, ...] if output is not None else None,
            w[0, ...] if w is not None else None,
            linear_output[0, ...] if linear_output is not None else None,
            linear_w[0, ...] if linear_w is not None else None
        )
        fig.suptitle(title, size="medium")
        writer.add_fig("output_slice_dur", fig)

        fig, _ = self._plot_velocity(
            target_output[0, ...],
            output[0, ...] if output is not None else None,
            linear_output[0, ...] if linear_output is not None else None,
        )
        fig.suptitle(title, size="medium")
        writer.add_fig("velocity_slice_dur", fig)

        if isinstance(model, ModalSystem):
            model.dur = self._gen.dur
            pred = model(
                y0[0, ...].unsqueeze(0),
                omega[0, ...].unsqueeze(0),
                sigma[0, ...].unsqueeze(0),
                gamma[0].unsqueeze(0),
                xe[0].unsqueeze(0),
                xo[0].unsqueeze(0),
                exc_amp[0].unsqueeze(0) if exc_amp is not None else None,
                exc_dur[0].unsqueeze(0) if exc_dur is not None else None,
                exc_st[0].unsqueeze(0) if exc_st is not None else None,
                exc_type[0].unsqueeze(0) if exc_type is not None else None
            )
            xlim = [
                1,
                2.0 * np.pi * NUM_PERIODS / omega[0, 0].item() * model.fs
            ]
            instance = copy.deepcopy(self[idx[0].item()])
            instance.update(self._getlin(idx[0].item()))

            fig, _ = self._plot_output(
                instance["target_output"],
                instance["target_w"],
                pred["output"][0, ...],
                pred["w"][0, ...],
                instance["linear_output"],
                instance["linear_w"],
                xlim
            )
            fig.suptitle(title, size="medium")
            writer.add_fig("output_dataset_dur", fig)

            fig, _ = self._plot_velocity(
                instance["target_output"],
                pred["output"][0, ...],
                instance["linear_output"],
                xlim
            )
            fig.suptitle(title, size="medium")
            writer.add_fig("velocity_dataset_dur", fig)

            writer.add_audio("wave_dataset_dur", pred["w"][0, ...], fs=model.fs)

            if self._gen.num_modes <= 2:
                fig, _ = self._plot_nonlinearity(self._gen.nl, model.nl, target_output)
                writer.add_fig("nonlinearity", fig)

            self._log_activations(writer, model.nl, target_output)

    @staticmethod
    def _log_activations(writer: WandBWriter, nl: nn.Module, target_output: torch.Tensor):
        if not isinstance(nl, torch.jit.ScriptModule):
            nl = nl.cpu()
            num_modes = target_output.shape[1] // 2
            q_points = target_output[:, :num_modes, :].detach().cpu()
            q_points = torch.movedim(q_points, -1, 1)
            q_points = torch.reshape(q_points, (-1, *q_points.shape[2:]))

            hooks = dict()
            def activation_hook(module, input, output, name):
                histogram_name = name + "_" + repr(module)
                writer.add_histogram(histogram_name, output)
                hooks[name].remove()
                del hooks[name]

            for name, module in nl.named_modules(remove_duplicate=False):
                if isinstance(module, nn.Sequential) or (module == nl):
                    continue
                hooks[name] = module.register_forward_hook(partial(activation_hook, name=name))

            nl(q_points)

    @staticmethod
    def _plot_instance(
            axs,
            target_w: torch.Tensor,
            fs: int,
            exc_amp: float = None,
            exc_dur: float = None,
            exc_st: float = None,
            exc_type: int = None,
            linear_w: torch.Tensor = None,
            **instance
        ):
        target_w = target_w.detach().cpu().numpy()
        if linear_w is not None:
            linear_w = linear_w.detach().cpu().numpy()
        
        num_samples = target_w.shape[-1]
        t_points = np.arange(start=0, stop=num_samples, step=1) / fs

        p = axs[0].plot(t_points, target_w)
        color = p[0].get_color()
        if linear_w is not None:
            axs[0].plot(t_points, linear_w, color=color, linestyle="dashed")

        fe = Excitation(
            torch.as_tensor(exc_amp),
            torch.as_tensor(exc_dur),
            torch.as_tensor(exc_st),
            torch.as_tensor(exc_type)
        )
        fe_points = fe(torch.from_numpy(t_points)).squeeze().numpy()
        axs[1].plot(t_points, fe_points, color=color)

    @staticmethod
    def _plot_output(
            target_output: torch.Tensor,
            target_w: torch.Tensor,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
            xlim: list[float] = None
        ):
        target_output = target_output.detach().cpu().numpy()
        target_w = target_w.detach().cpu().numpy()
        if output is not None:
            output = output.detach().cpu().numpy()
        if w is not None:
            w = w.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()
        if linear_w is not None:
            linear_w = linear_w.detach().cpu().numpy()

        num_modes = target_output.shape[0] // 2
        num_plots = min(NUM_MODES, num_modes)
        fig, axs = plt.subplots(nrows=(num_plots + 1), ncols=1, layout="constrained")

        axs[0].set_ylabel("Output wave", fontsize="small")
        if linear_w is not None:
            axs[0].plot(linear_w, label="Linear", color="tab:blue")
        axs[0].plot(target_w, label="Target", color="tab:orange")
        if w is not None:
            axs[0].plot(w, label="Predicted", color="tab:green", linestyle="dashed")
        axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
        if xlim:
            axs[0].set_xlim(xlim)
        axs[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axs[0].yaxis.get_offset_text().set_fontsize("small")
        axs[0].tick_params(axis="both", labelsize="small")

        for n in range(num_plots):
            m = START_MODE + n
            axs[n + 1].set_ylabel(f"Mode {m + 1}", fontsize="small")
            if linear_output is not None:
                axs[n + 1].plot(linear_output[m, :], label="Linear", color="tab:blue")
            axs[n + 1].plot(target_output[m, :], label="Target", color="tab:orange")
            if output is not None:
                axs[n + 1].plot(output[m, :], label="Predicted", color="tab:green", linestyle="dashed")
            if xlim:
                axs[n + 1].set_xlim(xlim)
            axs[n + 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs[n + 1].yaxis.get_offset_text().set_fontsize("small")
            axs[n + 1].tick_params(axis="both", labelsize="small")
        axs[n + 1].set_xlabel("Sample [n]", fontsize="small")

        return fig, axs

    @staticmethod
    def _plot_velocity(
            target_output: torch.Tensor,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            xlim: list[float] = None
        ):
        target_output = target_output.detach().cpu().numpy()
        if output is not None:
            output = output.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()

        num_modes = target_output.shape[0] // 2
        num_plots = min(NUM_MODES, num_modes)
        fig, axs = plt.subplots(nrows=num_plots, ncols=1, layout="constrained", squeeze=False)

        for n in range(num_plots):
            m = START_MODE + n
            axs[n, 0].set_ylabel(f"Velocity {m + 1}", fontsize="small")
            if linear_output is not None:
                axs[n, 0].plot(linear_output[num_modes + m, :], label="Linear", color="tab:blue")
            axs[n, 0].plot(target_output[num_modes + m, :], label="Target", color="tab:orange")
            if output is not None:
                axs[n, 0].plot(output[num_modes + m, :], label="Predicted", color="tab:green", linestyle="dashed")
            if xlim:
                axs[n, 0].set_xlim(xlim)
            axs[n, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs[n, 0].yaxis.get_offset_text().set_fontsize("small")
            axs[n, 0].tick_params(axis="both", labelsize="small")
        axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
        axs[n, 0].set_xlabel("Sample [n]", fontsize="small")

        return fig, axs

    @staticmethod
    def _plot_nonlinearity(target_nl: Callable, nl: nn.Module, target_output: torch.Tensor):
        nl = nl.cpu()
        target_output = target_output.detach().cpu()
        num_modes = target_output.shape[1] // 2

        q_points = torch.zeros((NUM_POINTS, num_modes), dtype=target_output.dtype)
        for m in range(num_modes):
            tmp = target_output[:, m, :].flatten()
            q_max = torch.max(tmp)
            q_min = torch.min(tmp)
            q_step = (q_max - q_min) / NUM_POINTS
            q_points[:, m] = torch.arange(start=q_min, end=q_max, step=q_step, dtype=q_points.dtype)

        if num_modes == 1:
            fig, axs = plt.subplots(nrows=1, ncols=1, layout="constrained")

            axs.plot(q_points.numpy(), target_nl(q_points).numpy(), label="Target")
            axs.plot(q_points.numpy(), nl(q_points).detach().numpy(), label="Predicted", linestyle="dashed")
            axs.set_xlabel("Displacement", fontsize="small")
            axs.set_ylabel("Output", fontsize="small")
            fig.legend(loc="outside upper center", ncols=2, fontsize="small")
            axs.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
            axs.xaxis.get_offset_text().set_fontsize("small")
            axs.yaxis.get_offset_text().set_fontsize("small")
            axs.tick_params(axis="both", labelsize="small")
            axs.grid()
        elif num_modes == 2:
            fig, axs = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, layout="constrained")

            x, y = torch.meshgrid(q_points[:, 0], q_points[:, 1], indexing="ij")
            input = torch.cat([x.reshape((-1, 1)), y.reshape((-1, 1))], dim=-1)
            nl_output = nl(input).detach().numpy()
            target_nl_output = target_nl(input).numpy()
            x = x.numpy()
            y = y.numpy()

            for m in range(num_modes):
                z = nl_output[:, m].reshape((NUM_POINTS, NUM_POINTS))
                target_z = target_nl_output[:, m].reshape((NUM_POINTS, NUM_POINTS))
                
                axs[m].plot_surface(x, y, target_z, alpha=0.5, rstride=100, cstride=100, edgecolors='k', label="Target")
                axs[m].plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, edgecolors='k', label="Predicted")
                axs[m].set_xlabel("Mode 1", fontsize="small")
                axs[m].set_ylabel("Mode 2", fontsize="small")
                axs[m].set_zlabel(f"Output {m + 1}", fontsize="small")
                axs[m].set_zlim(np.min(target_z), np.max(target_z))
                if m == 0:
                    fig.legend(loc="outside upper center", ncols=2, fontsize="small")
                axs[m].tick_params(axis="both", labelsize="small")
        else:
            print(f"Unsupported number of modes: {num_modes}")
            fig = None
            axs = None

        return fig, axs

    @property
    def md5(self):
        data = [
            self._gen.fs,
            self._gen.dur,
            self._gen.method,
            self._gen.num_modes,
            OmegaConf.to_object(self._gamma_range) if type(self._gamma_range) is ListConfig else self._gamma_range,
            OmegaConf.to_object(self._kappa_range) if type(self._kappa_range) is ListConfig else self._kappa_range,
            OmegaConf.to_object(self._sigma0_range) if type(self._sigma0_range) is ListConfig else self._sigma0_range,
            OmegaConf.to_object(self._sigma1_range) if type(self._sigma1_range) is ListConfig else self._sigma1_range,
            OmegaConf.to_object(self._xe_range) if type(self._xe_range) is ListConfig else self._xe_range,
            OmegaConf.to_object(self._xo_range) if type(self._xo_range) is ListConfig else self._xo_range,
            OmegaConf.to_object(self._exc_amp_range) if type(self._exc_amp_range) is ListConfig else self._exc_amp_range,
            OmegaConf.to_object(self._exc_dur_range) if type(self._exc_dur_range) is ListConfig else self._exc_dur_range,
            self._exc_st,
            self._exc_type,
            self._num_variations
        ]
        return calc_md5(data)
