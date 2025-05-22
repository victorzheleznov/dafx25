from collections.abc import Callable
from math import floor, sqrt

import torch
from torch import nn

from src.utils.excitation import Excitation
from src.utils.modal_solver import ModalIVP, ModalVerlet

USE_JIT = False

SOLVERS: dict[str, nn.Module] = {
    "verlet": ModalVerlet
}


class ModalSystem(nn.Module):
    """Class for physical or physics-informed modelling using modal synthesis."""
    def __init__(
            self,
            fs: int,
            dur: float,
            method: str,
            nl: Callable | nn.Module
        ):
        """Parameters
        ----------
        fs : int
            Sampling rate [Hz].
        dur : float
            Simulation duration [sec].
        method : str
            Simulation method (as defined in `SOLVERS` dictionary).
        nl : Callable | nn.Module
            Nonlinear function that describes coupling between the modes (can be either a target nonlinearity or 
            a neural network used for training).
        """
        super().__init__()

        self._fs = int(fs)
        self._dur = dur
        self._update()

        self._fe = Excitation()

        self._method = method
        self._solver = SOLVERS[self._method](nl=nl)
        if USE_JIT:
            self._solver = torch.jit.script(self._solver)

    def _update(self):
        self._num_samples = floor(self._dur * self._fs)

    @staticmethod
    def _Phi(x: torch.Tensor, num_modes: int) -> torch.Tensor:
        beta = torch.arange(start=1, end=(num_modes + 1), step=1, device=x.device) * torch.pi
        return sqrt(2) * torch.sin(torch.outer(x, beta))

    def _calc_exc(
            self,
            exc_amp: torch.Tensor,
            exc_dur: torch.Tensor,
            exc_st: torch.Tensor,
            exc_type: torch.Tensor,
            device: torch.device = None
        ) -> torch.Tensor:
        t_points = torch.arange(start=0, end=self._num_samples, step=1, device=device) / self._fs
        self._fe.amp = exc_amp
        self._fe.dur = exc_dur
        self._fe.st = exc_st
        self._fe.type = exc_type
        return self._fe(t_points)

    def forward(
            self,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            xe: torch.Tensor,
            xo: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None,
            **batch
        ):
        """Synthesise model output for given initial conditions, modal and excitation parameters.

        Parameters
        ----------
        y0 : torch.Tensor
            Initial conditions (batch size, state dimension).
        omega : torch.Tensor
            Modal angular frequencies (batch size, number of modes).
        sigma : torch.Tensor
            Modal damping parameters (batch size, number of modes).
        gamma : torch.Tensor
            Scaling parameters for nonlinear function (batch size,).
        xe : torch.Tensor
            Excitation positions (batch size,).
        xo : torch.Tensor
            Output positions (batch size,).
        exc_amp : torch.Tensor
            Excitation amplitudes (batch size,).
        exc_dur : torch.Tensor
            Excitation durations (batch size,).
        exc_st : torch.Tensor
            Excitation starting times (batch size,).
        exc_type : torch.Tensor
            Excitation types (batch size,).

        Returns
        -------
        out : dict[str, torch.Tensor]
            Computed system state ("output" key) and audio output ("w" key).
        """
        num_modes = omega.shape[-1]
        Phi_e = self._Phi(xe, num_modes)
        Phi_o = self._Phi(xo, num_modes)
        fe_points = self._calc_exc(exc_amp, exc_dur, exc_st, exc_type, device=y0.device)
        return self._solver(
            ModalIVP(
                self._fs,
                self._num_samples,
                y0,
                omega,
                sigma,
                gamma,
                Phi_e,
                Phi_o,
                fe_points
            )
        )

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value: int | float):
        if value is not None:
            self._fs = int(value)
            self._update()

    @property
    def dur(self):
        return self._dur

    @dur.setter
    def dur(self, value: float):
        if value is not None:
            self._dur = value
            self._update()

    @property
    def method(self):
        return self._method

    @property
    def nl(self):
        return self._solver._nl
