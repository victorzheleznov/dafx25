from pathlib import Path

import hydra
import matplotlib
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.data import get_dataloaders
from src.utils.init import set_random_seed, setup_logging, setup_saving
from src.utils.trainer import Trainer

matplotlib.use('Agg', force=False)


@hydra.main(version_base=None, config_path="cfg", config_name="nonlinear_oscillator")
def main(cfg):
    set_random_seed(cfg.seed)
    cfg_dir = Path([path["path"] for path in HydraConfig.get().runtime.config_sources if path["provider"] == "main"][0])
    cfg_name = HydraConfig.get().job.config_name
    save_dir, resume = setup_saving(cfg, cfg_dir, cfg_name)
    setup_logging(save_dir, resume)

    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    writer = instantiate(cfg.writer, run_cfg)

    print(f"Loaded {save_dir}" if resume else f"Created {save_dir}")

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
 
    dataset = instantiate(cfg.dataset)
    collate_func = instantiate(cfg.collate_func)
    dataloaders = get_dataloaders(dataset, cfg.split, cfg.batch_size, collate_func)

    loss_func = instantiate(cfg.loss_func).to(torch.double).to(device)
    model = instantiate(cfg.model).to(torch.double).to(device)
    optimiser = instantiate(cfg.optimiser, params=model.parameters())

    trainer = Trainer(
        model,
        loss_func,
        optimiser,
        device,
        dataloaders,
        cfg.monitor,
        save_dir,
        cfg.early_stop,
        cfg.log_step,
        cfg.max_grad_norm,
        writer,
        resume
    )
    trainer.train(cfg.num_epochs)

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()