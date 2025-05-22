from collections.abc import Callable

import torch
from torch import nn


class ModalIVP:
    """Class defining initial value problem for numerical solvers."""
    def __init__(
            self,
            fs: int,
            num_samples: int,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            Phi_e: torch.Tensor,
            Phi_o: torch.Tensor,
            fe_points: torch.Tensor
        ):
        """Parameters
        ----------
        fs : int
            Sampling rate [Hz].
        num_samples : int
            Number of samples (i.e., integration steps).
        y0 : torch.Tensor
            Initial conditions (batch size, state dimension).
        omega : torch.Tensor
            Modal angular frequencies (batch size, number of modes).
        sigma : torch.Tensor
            Modal damping parameters (batch size, number of modes).
        gamma : torch.Tensor
            Scaling parameters for nonlinear function (batch size,).
        Phi_e : torch.Tensor
            Modal shapes for excitation positions (batch size, number of modes).
        Phi_o : torch.Tensor
            Modal shapes for output positions (batch size, number of modes).
        fe_points : torch.Tensor
            Precomputed excitation profiles (batch size, number of samples).
        """
        self.fs = fs
        self.num_samples = num_samples
        self.y0 = y0
        self.fe_points = fe_points
        self.omega = omega
        self.sigma = sigma
        self.gamma = gamma
        self.Phi_e = Phi_e
        self.Phi_o = Phi_o

        if not torch.jit.is_scripting():
            assert y0.ndim == 2
            assert omega.ndim == 2
            assert sigma.ndim == 2
            assert gamma.ndim == 1
            assert Phi_e.ndim == 2
            assert Phi_o.ndim == 2
            assert fe_points.ndim == 2

            num_modes = omega.shape[-1]
            assert y0.shape[-1] == int(2 * num_modes)
            assert sigma.shape[-1] == num_modes
            assert Phi_e.shape[-1] == num_modes
            assert Phi_o.shape[-1] == num_modes
            assert fe_points.shape[-1] == num_samples


class ModalVerlet(nn.Module):
    """Class for the Stormer-Verlet numerical method for modal synthesis."""
    def __init__(self, nl: Callable | nn.Module):
        """Parameters
        ----------
        nl : Callable | nn.Module
            Nonlinear function that describes coupling between the modes (can be either a target nonlinearity or 
            a neural network used for training).
        """
        super().__init__()
        self._nl = nl

    def forward(self, ivp: ModalIVP) -> dict[str, torch.Tensor]:
        """Solve given initial value problem.

        Parameters
        ----------
        ivp : ModalIVP
            Initial value problem.

        Returns
        -------
        out : dict[str, torch.Tensor]
            Computed system state ("output" key) and audio output ("w" key).
        """
        # precompute numerical method parameters
        k = 1.0 / ivp.fs
        d = 1.0 + k * ivp.sigma

        # parse initial conditions
        num_modes = ivp.omega.shape[-1]
        q0 = ivp.y0[:, :num_modes]
        p0 = ivp.y0[:, num_modes:]
        w0 = (ivp.Phi_o * q0).sum(-1)

        # initialise
        y = torch.zeros((ivp.y0.shape[0], ivp.y0.shape[1], ivp.num_samples), dtype=ivp.y0.dtype, device=ivp.y0.device)
        w = torch.zeros((ivp.y0.shape[0], ivp.num_samples), dtype=ivp.y0.dtype, device=ivp.y0.device)
        y[..., 0] = ivp.y0
        w[:, 0] = w0

        # main loop
        for n in range(ivp.num_samples - 1):
            # update state
            fe0 = ivp.fe_points[:, n]
            fe1 = ivp.fe_points[:, n + 1]
            nl0 = self._nl(q0)
            p_dot = -2.0 * ivp.sigma * p0 - ivp.omega**2 * q0 + (ivp.gamma**2).unsqueeze(-1) * nl0 + fe0.unsqueeze(-1) * ivp.Phi_e
            p_half = p0 + k / 2 * p_dot
            q1 = q0 + k * p_half
            nl1 = self._nl(q1)
            p1 = (p_half + k / 2 * (-ivp.omega**2 * q1 + (ivp.gamma**2).unsqueeze(-1) * nl1 + fe1.unsqueeze(-1) * ivp.Phi_e)) / d
            y1 = torch.cat([q1, p1], dim=-1)
            w1 = (ivp.Phi_o * q1).sum(-1)

            # shift state
            y[..., n + 1] = y1
            w[:, n + 1] = w1
            q0 = q1
            p0 = p1

        return {"output": y, "w": w}
