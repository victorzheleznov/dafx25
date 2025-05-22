import torch
from torch import nn

from src.losses import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self, tensor_name: str = "output"):
        super().__init__(tensor_name)
        self._loss = nn.MSELoss()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        mse_loss = self._loss(output, target_output)
        return {"loss": mse_loss, "mse_loss": mse_loss}
