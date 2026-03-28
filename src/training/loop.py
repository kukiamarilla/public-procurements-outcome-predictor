from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from training.early_stopping import EarlyStopping


def train_one_fold(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: CFG,
    fold: Optional[int] = None,
    patience: int = 5,
    bce_pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, list], float, int]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        if bce_pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        restore_best=True,
    )

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    fold_str = f"[Fold {fold}] " if fold is not None else ""

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_total = 0.0

        for embs, valid, y in train_dl:
            embs = embs.to(cfg.device)
            valid = valid.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(embs, valid)
            loss: torch.Tensor = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_total += float(loss.item())

        train_loss = train_total / max(1, len(train_dl))

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for embs, valid, y in val_dl:
                embs = embs.to(cfg.device)
                valid = valid.to(cfg.device)
                y = y.to(cfg.device)

                logits = model(embs, valid)
                loss = loss_fn(logits, y)
                val_total += float(loss.item())

        val_loss = val_total / max(1, len(val_dl))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"{fold_str}epoch {ep:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}",
        )

        early_stopping(val_loss, model, ep)
        if early_stopping.early_stop:
            print(f"{fold_str}Early stopping en epoch {ep}")
            break

    early_stopping.restore(model)
    return history, early_stopping.best_loss, early_stopping.best_epoch


@torch.no_grad()
def evaluate_probs(
    model: nn.Module,
    dl: DataLoader,
    cfg: CFG,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_probs: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    for embs, valid, y in dl:
        embs = embs.to(cfg.device)
        valid = valid.to(cfg.device)

        logits = model(embs, valid)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_y.append(y.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_y, dim=0)
