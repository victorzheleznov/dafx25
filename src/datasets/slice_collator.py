from math import floor

import torch
from torch.utils.data import default_collate


class SliceCollator:
    def __init__(self, slice_dur: float, overlap: bool=False):
        self._slice_dur = slice_dur
        self._overlap = overlap

    def __call__(self, instances: list[dict]) -> dict:
        # construct default batch
        default_batch = default_collate(instances)
        keys = list(default_batch.keys())
        batch = dict()

        # get sample rate and number of samples (assumed the same for all instances!)
        assert torch.equal(default_batch["fs"], default_batch["fs"][0].item() * torch.ones_like(default_batch["fs"])), "All sample rates should be the same!"
        assert torch.equal(default_batch["dur"], default_batch["dur"][0].item() * torch.ones_like(default_batch["dur"])), "All durations should be the same!"
        fs = int(instances[0]["fs"])
        num_samples = instances[0]["target_output"].shape[-1]

        # get sizes
        slice_len = floor(self._slice_dur * instances[0]["fs"])
        if self._overlap:
            step = slice_len - 1
        else:
            step = slice_len
        num_slices = (num_samples - slice_len) // step + 1

        # slice waveforms
        for key in keys:
            if ("target" in key) or ("linear" in key) or ("output" in key): # (batch_size, ..., num_samples)
                b = default_batch.pop(key).unfold(-1, slice_len, step)      # (batch_size, ..., num_slices, slice_len)
                b = torch.movedim(b, -2, 1)                                 # (batch_size, num_slices, ..., slice_len)
                b = torch.reshape(b, (-1, *b.shape[2:]))                    # (batch_size, ..., slice_len)
                batch[key] = b

        # update initial conditions
        batch["y0"] = batch["target_output"][..., 0]
        default_batch.pop("y0")

        # adjust excitation
        if "exc_st" in default_batch.keys():
            batch["exc_st"] = self._adjust_time(default_batch.pop("exc_st"), step, num_slices, fs)

        # adjust timing
        batch["fs"] = self._hold(default_batch.pop("fs"), num_slices)
        batch["dur"] = slice_len / batch["fs"]
        default_batch.pop("dur")

        # hold values for all other keys
        for key in default_batch.keys():
            batch[key] = self._hold(default_batch[key], num_slices)

        return batch

    @staticmethod
    def _hold(data: list | torch.Tensor, length: int):
        if type(data) is list:
            out = [data[i] for i in range(len(data)) for _ in range(length)]
        elif type(data) is torch.Tensor:
            out = torch.flatten(data.unsqueeze(1) * torch.ones(data.shape[:1] + (length,) + data.shape[1:], dtype=data.dtype, device=data.device), start_dim=0, end_dim=1)
        else:
            raise NotImplementedError()
        return out

    @staticmethod
    def _adjust_time(t_points: torch.Tensor, step: int, num_slices: int, fs: int):
        idx = torch.arange(start=0, end=num_slices, step=1)
        t_st = idx * step / fs
        return torch.flatten(t_points.unsqueeze(-1) - t_st.unsqueeze(0))
