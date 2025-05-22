import hashlib
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image


def calc_md5(data: list):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def get_image(fig: plt.Figure):
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return img


def gen_from_range(
        range: list[float],
        shape: tuple[int, ...],
        dtype: torch.dtype = None,
        device: torch.device = None
    ) -> torch.Tensor:
    if range[0] == range[1]:
        out = range[0] * torch.ones(shape, dtype=dtype, device=device)
    else:
        out = torch.distributions.uniform.Uniform(
            range[0],
            range[1]
        ).sample(shape).to(dtype).to(device)
    return out
