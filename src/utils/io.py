from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def write_wav(file: Path | str, data: np.ndarray | torch.Tensor, fs: int, normalise=False):
    """Write audio to a .wav file"""
    file = Path(file)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if normalise:
        data = data / np.max(np.abs(data))
    sf.write(file, data, int(fs), "DOUBLE")