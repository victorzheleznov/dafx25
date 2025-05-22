import os
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.utils.data import get_dataloaders
from src.utils.init import load_env, set_random_seed

matplotlib.use('TkAgg', force=False)
load_env()


def move_batch_to_device(batch: dict, device: str):
    for key in batch.keys():
        if type(batch[key]) is torch.Tensor:
            batch[key] = batch[key].to(device)
    return batch


@hydra.main(version_base=None, config_path=os.environ["PYTHON_LAST_SAVE_DIR"], config_name="cfg")
def main(cfg):
    set_random_seed(cfg.seed)
    save_dir = Path([path["path"] for path in HydraConfig.get().runtime.config_sources if path["provider"] == "main"][0])
    print(f"Loaded {save_dir}")

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    dataset = instantiate(cfg.dataset)
    dataloader = get_dataloaders(dataset, cfg.split, cfg.batch_size)[cfg.partition]
    if ("dur" in cfg.model) and ("dur" in cfg.dataset):
        cfg.model.dur = cfg.dataset.dur
    if ("fs" in cfg.model) and ("fs" in cfg.dataset):
        cfg.model.fs = cfg.dataset.fs

    if hasattr(dataset, "save_pred"):
        if callable(dataset.save_pred):
            test_dir = Path("test") / save_dir.stem / str(dataset) / cfg.partition
            test_dir.mkdir(exist_ok=True, parents=True)

    loss_func = instantiate(cfg.loss_func).to(torch.double).to(device)
    model = instantiate(cfg.model).to(torch.double).to(device)
    
    model.load_state_dict(torch.load(save_dir / "checkpoint_best.pt", weights_only=True)["model_state_dict"])

    with torch.no_grad():
        model.eval()
        running_losses = defaultdict(lambda: 0.0)
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            pred = model(**batch)
            batch.update(pred)
            losses = loss_func(batch)
            for key in losses.keys():
                running_losses[key] += losses[key].item()
            if hasattr(dataset, "save_pred"):
                if callable(dataset.save_pred):
                    dataset.save_pred(test_dir, **batch)
        loss = running_losses["loss"] / len(dataloader)
        print(f"Loss for {cfg.partition} dataset: {loss}")
        if __debug__:
            batch.update({"model": model})
            batch.update({"key": cfg.partition})
            dataset.plot_batch(**batch)
            plt.show()


if __name__ == "__main__":
    main()