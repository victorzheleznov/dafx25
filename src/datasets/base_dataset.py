from torch.utils.data import Dataset

from src.utils.writer import WandBWriter


class BaseDataset(Dataset):
    def plot_batch(self, **batch):
        raise NotImplementedError("Subclasses of BaseDataset should implement plot_batch.")

    def log_batch(self, writer: WandBWriter, **batch):
        raise NotImplementedError("Subclasses of BaseDataset should implement log_batch.")
