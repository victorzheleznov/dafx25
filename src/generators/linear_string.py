import torch

from src.models import ModalSystem
from src.utils.func import ZeroFunc


class LinearString(ModalSystem):
    def __init__(
            self,
            fs: int,
            dur: float,
            method: str,
            num_modes: int
        ):
        nl = ZeroFunc()

        super().__init__(
            fs,
            dur,
            method,
            nl
        )

        self._num_modes = num_modes
        self._omega = None
        self._sigma = None

    @staticmethod
    def _calc_modes(
            gamma: torch.Tensor,
            kappa: torch.Tensor,
            sigma0: torch.Tensor,
            sigma1: torch.Tensor,
            num_modes: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.arange(start=1, end=(num_modes + 1), step=1, device=gamma.device) * torch.pi
        omega = torch.sqrt(torch.outer(gamma**2, beta**2) + torch.outer(kappa**2, beta**4))
        sigma = sigma0.unsqueeze(-1) + torch.outer(sigma1, beta**2)
        return omega, sigma

    @torch.no_grad()
    def forward(
            self,
            y0: torch.Tensor,
            gamma: torch.Tensor,
            kappa: torch.Tensor,
            sigma0: torch.Tensor,
            sigma1: torch.Tensor,
            xe: torch.Tensor,
            xo: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None
        ):
        self._omega, self._sigma = self._calc_modes(gamma, kappa, sigma0, sigma1, self._num_modes)
        return super().forward(
            y0,
            self._omega,
            self._sigma,
            gamma,
            xe,
            xo,
            exc_amp=exc_amp,
            exc_dur=exc_dur,
            exc_st=exc_st,
            exc_type=exc_type
        )

    @property
    def num_modes(self):
        return self._num_modes

    @property
    def omega(self):
        return self._omega

    @property
    def sigma(self):
        return self._sigma
