from __future__ import annotations

import copy

import torch.nn as nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float("inf")
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: nn.Module) -> None:
        if self.restore_best and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Modelo restaurado al mejor epoch: {self.best_epoch}")
