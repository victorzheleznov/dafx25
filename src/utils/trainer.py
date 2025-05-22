import glob
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from numpy import inf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.tracker import Tracker
from src.utils.writer import WandBWriter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            loss_func: nn.Module,
            optimiser: Optimizer,
            device: str,
            dataloaders: dict[DataLoader | None],
            monitor: str,
            save_dir: Path,
            early_stop: int = -1,
            log_step: int = -1,
            max_grad_norm: float = None,
            writer: WandBWriter = None,
            resume: bool = False
        ):
        self._model = model
        self._loss_func = loss_func
        self._optimiser = optimiser
        self._device = device
        self._train_dataloader = dataloaders["train"]
        self._valid_dataloader = dataloaders["valid"]
        self._save_dir = save_dir
        self._max_grad_norm = max_grad_norm
        self._writer = writer

        self._monitor_mode, self._monitor_section, self._monitor_metric = monitor.replace("/", " ").split()
        assert self._monitor_mode in ["min", "max"]
        self._monitor_best = inf if self._monitor_mode == "min" else -inf
        self._early_stop = early_stop if early_stop > 0 else inf
        self._log_step = log_step if log_step > 0 else inf

        self._tracker = Tracker()
        self._start_epoch = 1
        self._num_epochs = None
        if self._valid_dataloader is not None:
            self._dataset = self._valid_dataloader.dataset.dataset

        if resume:
            self._resume_checkpoint()

    def train(self, num_epochs: int):
        self._num_epochs = num_epochs
        try:
            self._train_process()
        except KeyboardInterrupt as error:
            self._save_checkpoint()
            if self._writer:
                self._writer.add_checkpoints(self._save_dir)
            raise error

    def _train_process(self):
        not_improved_count = 0
        for epoch in tqdm(range(self._start_epoch, self._start_epoch + self._num_epochs)):
            logger.info(f"Epoch {epoch} started...")
            self._tracker.update({"epoch": epoch})
            self._train_epoch(epoch)
            if self._valid_dataloader is not None:
                self._valid_epoch(epoch)
            not_improved_count, stop_process = self._monitor_performance(not_improved_count)
            if stop_process:
                break
        self._save_checkpoint()
        if self._writer:
            self._writer.add_checkpoints(self._save_dir)

    def _train_epoch(self, epoch: int):
        self._model.train()
        running_losses = defaultdict(lambda: 0.0)
        running_grad_norm = 0.0
        for batch in self._train_dataloader:
            self._optimiser.zero_grad()

            batch = self._move_batch_to_device(batch)
            pred = self._model(**batch)
            batch.update(pred)
            losses = self._loss_func(batch)
            losses["loss"].backward()
            self._clip_grad_norm()
            self._optimiser.step()

            for key in losses.keys():
                running_losses[key] += losses[key].item()
            running_grad_norm += self._get_grad_norm()
        for key in losses.keys():
            self._tracker["train"].update({key: running_losses[key] / len(self._train_dataloader)})
        self._tracker["train"].update({"grad_norm": running_grad_norm / len(self._train_dataloader)})
        logger.info(self._tracker.print("train"))
        if self._writer:
            self._writer.set_step(epoch, "train")
            self._writer.add_scalars(self._tracker["train"])

    @torch.no_grad()
    def _valid_epoch(self, epoch: int):
        self._model.eval()
        running_losses = defaultdict(lambda: 0.0)
        for batch in self._valid_dataloader:
            batch = self._move_batch_to_device(batch)
            pred = self._model(**batch)
            batch.update(pred)
            losses = self._loss_func(batch)
            for key in losses.keys():
                running_losses[key] += losses[key].item()
        for key in losses.keys():
            self._tracker["valid"].update({key: running_losses[key] / len(self._valid_dataloader)})
        logger.info(self._tracker.print("valid"))
        if self._writer:
            self._writer.set_step(epoch, "valid")
            self._writer.add_scalars(self._tracker["valid"])
            if (epoch % self._log_step == 0) or (epoch == self._start_epoch):
                self._log_batch(batch)

    def _log_batch(self, batch: dict):
        batch.update({"model": deepcopy(self._model)})
        self._dataset.log_batch(self._writer, **batch)

    def _move_batch_to_device(self, batch: dict):
        for key in batch.keys():
            if type(batch[key]) is torch.Tensor:
                batch[key] = batch[key].to(self._device)
        return batch

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: int = 2):
        parameters = self._model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        return total_norm.item()

    def _clip_grad_norm(self, norm_type: int = 2):
        if self._max_grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), self._max_grad_norm, norm_type)

    def _monitor_performance(self, not_improved_count: int):
        epoch = self._tracker["epoch"]
        metric_value = self._tracker[self._monitor_section][self._monitor_metric]

        if self._monitor_mode == "max":
            improved = (self._monitor_best < metric_value)
        else:
            improved = (self._monitor_best > metric_value)

        if improved:
            self._monitor_best = metric_value
            self._save_checkpoint(save_best=True)
            not_improved_count = 0
        else:
            not_improved_count += 1

        if not_improved_count >= self._early_stop:
            logger.info(f"Performance did not improve for {not_improved_count} epochs! Stopping training...")
            stop_process = True
        else:
            stop_process = False

        if self._writer:
            self._writer.set_step(epoch, self._monitor_section)
            self._writer.add_scalar("monitor", self._monitor_best)

        return not_improved_count, stop_process

    def _save_checkpoint(self, save_best: bool = False):
        epoch = self._tracker["epoch"]
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimiser_state_dict": self._optimiser.state_dict(),
            "monitor": self._monitor_best
        }

        if save_best:
            checkpoint_file = self._save_dir / "checkpoint_best.pt"
        else:
            checkpoint_file = self._save_dir / f"checkpoint_{epoch}.pt"
        torch.save(checkpoint, checkpoint_file)

    def _resume_checkpoint(self):
        checkpoint_names = glob.glob("checkpoint_*.pt", root_dir=self._save_dir)
        checkpoint_names.remove("checkpoint_best.pt")
        checkpoint_names.sort(key=self._get_checkpoint_number)

        if checkpoint_names:
            checkpoint_file = self._save_dir / checkpoint_names[-1]
        else:
            checkpoint_file = self._save_dir / "checkpoint_best.pt"

        print(f"Resuming checkpoint {checkpoint_file}")
        logger.info(f"Resuming checkpoint {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, self._device, weights_only=True)
        self._start_epoch = checkpoint["epoch"] + 1
        self._monitor_best = checkpoint["monitor"]
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    @staticmethod
    def _get_checkpoint_number(checkpoint_name: str) -> int:
        return int(checkpoint_name.removeprefix("checkpoint_").removesuffix(".pt"))
