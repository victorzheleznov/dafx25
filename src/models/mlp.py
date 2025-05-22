import copy
from functools import partial

import torch
from torch import nn


class BaseMLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: list[int],
            out_dim: int,
            activation: nn.Module,
            bias: bool = True,
            factor: float = 1.0,
            init_name: str = "xavier_uniform_",
            **init_kwargs
        ):
        super().__init__()

        self._model = nn.Sequential()
        for n in hidden_dim:
            self._model.append(nn.Linear(in_features=in_dim, out_features=n, bias=bias))
            self._model.append(copy.deepcopy(activation))
            in_dim = n
        self._model.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias))

        self._factor = factor
        self._init_func = self._get_init_func(activation, init_name, **init_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self._init_func(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _get_init_func(activation: nn.Module, name: str, **kwargs):
        if "xavier" in name:
            if type(activation) is nn.ReLU:
                kwargs.update({"gain": nn.init.calculate_gain("relu")})
            elif type(activation) is nn.LeakyReLU:
                kwargs.update({"gain": nn.init.calculate_gain("leaky_relu", param=activation.negative_slope)})
            elif type(activation) is nn.Tanh:
                kwargs.update({"gain": nn.init.calculate_gain("tanh")})
            elif type(activation) is nn.Sigmoid:
                kwargs.update({"gain": nn.init.calculate_gain("sigmoid")})
        elif "kaiming" in name:
            if type(activation) is nn.ReLU:
                kwargs.update({"nonlinearity": "relu"})
            elif type(activation) is nn.LeakyReLU:
                kwargs.update({"nonlinearity": "leaky_relu", "a": activation.negative_slope})

        func = getattr(nn.init, name)
        func = partial(func, **kwargs)

        return func

    def forward(self, x: torch.Tensor):
        return self._model(self._factor * x)
