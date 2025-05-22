from collections.abc import Callable

import torch

from src.models import ModalSystem
from src.utils.func import PowerFunc, SinhFunc, TanhFunc, ZeroFunc

FUNCTIONS: dict[str, Callable] = {
    "power": PowerFunc,
    "tanh": TanhFunc,
    "sinh": SinhFunc,
    "zero": ZeroFunc
}


class NonlinearOscillator(ModalSystem):
    def __init__(
            self,
            fs: int,
            dur: float,
            method: str,
            nl_name: str = "power",
            **nl_kwargs
        ):
        nl = FUNCTIONS[nl_name](**nl_kwargs)

        super().__init__(
            fs,
            dur,
            method,
            nl
        )

        self.register_buffer("_xe", None, persistent=False)
        self.register_buffer("_xo", None, persistent=False)
        self._xe = torch.as_tensor([0.25])
        self._xo = torch.as_tensor([0.25])

    @torch.no_grad()
    def forward(
            self,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None
        ):
        return super().forward(
            y0,
            omega,
            sigma,
            gamma,
            self._xe,
            self._xo,
            exc_amp=exc_amp,
            exc_dur=exc_dur,
            exc_st=exc_st,
            exc_type=exc_type
        )

    @property
    def xe(self):
        return self._xe

    @property
    def xo(self):
        return self._xo