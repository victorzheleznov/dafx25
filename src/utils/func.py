import torch


class PowerFunc:
    def __init__(self, factor: float = -1.0, degree: int = 3):
        self._factor = factor
        self._degree = degree

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * x**self._degree


class ZeroFunc:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.0 * x


class TanhFunc:
    def __init__(self, factor: float = -1.0):
        self._factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.tanh(x)


class SinhFunc:
    def __init__(self, factor: float = -1.0):
        self._factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.sinh(x)
