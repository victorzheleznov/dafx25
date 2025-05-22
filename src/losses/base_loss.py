import torch
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, tensor_name: str = "output"):
        super().__init__()
        self._tensor_name = tensor_name

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        raise NotImplementedError

    def _parse_batch(self, batch: dict) -> tuple[torch.Tensor]:
        return batch[self._tensor_name], batch["target_" + self._tensor_name]
