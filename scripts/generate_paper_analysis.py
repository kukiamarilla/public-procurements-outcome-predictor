"""
Generate dataset summary statistics and scientifically interpretable figures for the short paper.

Outputs:
  - figures/pr_curve_cv.png
  - figures/calibration_curve_cv.png
  - figures/roc_curve_cv.png
  - outputs/paper_analysis_stats.json

The script reuses the same CV protocol and model used in train_cv_mlflow.py so that
the plotted predictions are out-of-fold predictions from the same experimental setup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import default_cfg
from data.chunk_dataset import (
    CachedChunkEmbDataset,
    collate_pad_chunks,
    list_labeled_embedding_paths_from_dataset_json,
)
from models.predictor import build_model_from_sample_batch
from training.loop import evaluate_probs, train_one_fold
from training.metrics import binary_classification_metrics
from training.reproducibility import configure_reproducibility, make_dataloader_worker_init_fn


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready figures and stats from CV predictions.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _fold_rng_seed(global_seed: int, fold: int) -> int:
    return int(global_seed) + int(fold) * 1_000_003


def _use_stratified(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2 and bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


def _compute_chunk_stats(paths: list[str]) -> dict[str, float]:
    chunk_counts: list[int] = []
    dims: list[int] = []
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        embs = d["embs"]
        chunk_counts.append(int(embs.shape[0]))
        dims.append(int(embs.shape[1]))
    arr = np.asarray(chunk_counts, dtype=np.float64)
    dims_arr = np.asarray(dims, dtype=np.float64)
    return {
        "n_samples": int(len(paths)),
        "mean_chunks": float(arr.mean()),
        "std_chunks": float(arr.std()),
        "median_chunks": float(np.median(arr)),
        "min_chunks": float(arr.min()),
        "max_chunks": float(arr.max()),
        "embedding_dim": float(dims_arr[0]) if len(dims_arr) else float("nan"),
    }


def _interp_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_ord = x[order]
    y_ord = y[order]
    x_unique, unique_idx = np.unique(x_ord, return_index=True)
    y_unique = y_ord[unique_idx]
    return np.interp(grid, x_unique, y_unique)


def main() -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install train dependencies first: uv sync --extra train") from e

    load_dotenv()
    args = _parse_args()
    configure_reproducibility(args.seed)

    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_json = args.dataset_json.expanduser().resolve()
    paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dataset_json))
    if len(paths) < args.folds:
        raise SystemExit(f"Need at least {args.folds} samples, found {len(paths)}")

    y = np.asarray(y_list, dtype=np.float64)
    y_int = y.astype(np.int64)

    cfg = default_cfg()
    cfg.cache_dir = str(cache_dir)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr

    indices = np.arange(len(paths))
    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(indices, y_int)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(indices)

    base_ds = CachedChunkEmbDataset(files=paths, y_list=y_list)

    oof_probs = np.full(len(paths), np.nan, dtype=np.float64)
    oof_y = np.full(len(paths), np.nan, dtype=np.float64)
    pr_curves: list[np.ndarray] = []
    roc_curves: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    pr_grid = np.linspace(0.0, 1.0, 201)
    roc_grid = np.linspace(0.0, 1.0, 201)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        fold_seed = _fold_rng_seed(args.seed, fold)
        configure_reproducibility(fold_seed)

        train_ds = Subset(base_ds, tr_idx.tolist())
        val_ds = Subset(base_ds, va_idx.tolist())

        dl_gen = torch.Generator()
        dl_gen.manual_seed(fold_seed + 9)
        worker_init = make_dataloader_worker_init_fn(fold_seed + 99) if args.num_workers > 0 else None

        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_pad_chunks,
            num_workers=args.num_workers,
            pin_memory=str(cfg.device).startswith("cuda"),
            generator=dl_gen,
            worker_init_fn=worker_init,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_pad_chunks,
            num_workers=args.num_workers,
            pin_memory=str(cfg.device).startswith("cuda"),
            worker_init_fn=worker_init,
        )

        sample_batch = cast(List[Tuple[torch.Tensor, torch.Tensor]], [train_ds[0]])
        embs0, _, _ = collate_pad_chunks(sample_batch)
        model = build_model_from_sample_batch(embs0, cfg)

        train_one_fold(model, train_dl, val_dl, cfg, fold=fold, patience=args.patience)
        probs_t, y_t = evaluate_probs(model, val_dl, cfg)
        probs = probs_t.numpy().reshape(-1)
        y_true = y_t.numpy().reshape(-1)

        oof_probs[va_idx] = probs
        oof_y[va_idx] = y_true

        metrics = binary_classification_metrics(probs, y_true, threshold=args.class_threshold)
        fold_metrics.append({k: float(v) for k, v in metrics.items() if isinstance(v, float) and v == v})

        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_curves.append(_interp_curve(recall[::-1], precision[::-1], pr_grid))

        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_curves.append(_interp_curve(fpr, tpr, roc_grid))

    mask = ~np.isnan(oof_probs)
    oof_probs = oof_probs[mask]
    oof_y = oof_y[mask]

    chunk_stats = _compute_chunk_stats(paths)
    chunk_stats["n_positive"] = int(np.sum(y > 0.5))
    chunk_stats["n_negative"] = int(np.sum(y <= 0.5))
    chunk_stats["positive_prevalence"] = float(np.mean(y))

    metrics_all = binary_classification_metrics(oof_probs, oof_y, threshold=args.class_threshold)

    out_stats = {
        "dataset": chunk_stats,
        "oof_metrics": metrics_all,
        "fold_metrics": fold_metrics,
        "folds": int(args.folds),
        "seed": int(args.seed),
        "source_dataset_json": str(dataset_json),
        "cache_dir": str(cache_dir),
    }

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_analysis_stats.json").write_text(json.dumps(out_stats, indent=2), encoding="utf-8")

    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Precision-recall curve
    pr_arr = np.vstack(pr_curves)
    pr_mean = pr_arr.mean(axis=0)
    pr_std = pr_arr.std(axis=0)
    prevalence = float(np.mean(oof_y))

    plt.figure(figsize=(5.2, 4.0))
    plt.plot(pr_grid, pr_mean, label="Mean PR curve", color="tab:blue", linewidth=2.0)
    plt.fill_between(
        pr_grid,
        np.clip(pr_mean - pr_std, 0.0, 1.0),
        np.clip(pr_mean + pr_std, 0.0, 1.0),
        color="tab:blue",
        alpha=0.18,
        label="Fold std.",
    )
    plt.axhline(prevalence, color="tab:red", linestyle="--", linewidth=1.5, label="Positive prevalence")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Across CV Folds")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve_cv.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # Calibration curve / reliability diagram using out-of-fold predictions
    frac_pos, mean_pred = calibration_curve(oof_y, oof_probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.2, 4.0))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", color="tab:green", linewidth=2.0, label="OOF calibration")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency of positives")
    plt.title("Calibration Curve (Out-of-Fold Predictions)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve_cv.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # ROC curve
    roc_arr = np.vstack(roc_curves)
    roc_mean = roc_arr.mean(axis=0)
    roc_std = roc_arr.std(axis=0)
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(roc_grid, roc_mean, label="Mean ROC curve", color="tab:purple", linewidth=2.0)
    plt.fill_between(
        roc_grid,
        np.clip(roc_mean - roc_std, 0.0, 1.0),
        np.clip(roc_mean + roc_std, 0.0, 1.0),
        color="tab:purple",
        alpha=0.18,
        label="Fold std.",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Chance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve Across CV Folds")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve_cv.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
