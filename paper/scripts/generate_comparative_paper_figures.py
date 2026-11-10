"""
Generate comparative paper figures for the strongest lexical and LLM-based models.

Outputs:
  - figures/pr_curve_cv.png
  - figures/calibration_curve_cv.png
  - figures/threshold_sensitivity_comparison.png
  - outputs/comparative_paper_figures_stats.json

The script recomputes out-of-fold probabilities with the same deterministic
5-fold protocol used elsewhere in the project so that the figures and paper
text remain contextually consistent.
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
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset, TensorDataset

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
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
from training.early_stopping import EarlyStopping
from training.loop import evaluate_probs, train_one_fold
from training.metrics import binary_classification_metrics
from training.reproducibility import configure_reproducibility, make_dataloader_worker_init_fn


class MeanPoolMLP(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparative figures for the paper.")
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
    p.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "pbc_txt_cache",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--mlp-hidden-dim", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.2)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--mlp-epochs", type=int, default=50)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--threshold-comparison-json",
        type=Path,
        default=PAPER_ROOT / "outputs" / "threshold_selection_comparison.json",
    )
    return p.parse_args()


def _fold_rng_seed(global_seed: int, fold: int) -> int:
    return int(global_seed) + int(fold) * 1_000_003


def _use_stratified(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2 and bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


def _aligned_tender_ids(paths: list[str]) -> list[str]:
    tids: list[str] = []
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        tid = str(d.get("tender_id") or "").strip()
        if not tid:
            raise ValueError(f"Missing tender_id in {path}")
        tids.append(tid)
    return tids


def _load_rows_by_tender_id(dataset_json_path: Path) -> dict[str, dict]:
    rows = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset-json must contain a list")
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId") or "").strip()
        if tid:
            out[tid] = row
    return out


def _safe_name(tid: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in tid)[:220]


def _load_cached_texts(
    *,
    tender_ids: list[str],
    rows_by_tid: dict[str, dict],
    cache_dir: Path,
) -> list[str]:
    texts: list[str] = []
    for tid in tender_ids:
        row = rows_by_tid.get(tid)
        if row is None:
            raise KeyError(f"TenderId not found in dataset-json: {tid}")
        txt_key = str(row.get("pbc_txt_s3_key") or "").strip()
        if not txt_key:
            raise ValueError(f"Missing pbc_txt_s3_key for tenderId={tid}")
        local_path = cache_dir / f"{_safe_name(tid)}.txt"
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Missing cached PBC text for {tid}: {local_path}. "
                f"Populate {REPO_ROOT / 'data' / 'raw' / 'pbc_txt_cache'} before generating paper figures."
            )
        texts.append(local_path.read_text(encoding="utf-8", errors="replace"))
    return texts


def _load_meanpooled_embeddings(paths: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        pooled = d["embs"].float().mean(dim=0)
        rows.append(pooled.numpy().astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def _train_eval_meanpool_mlp(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    d_in: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    fold_seed: int,
    num_workers: int,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_reproducibility(fold_seed)

    model = MeanPoolMLP(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=patience, min_delta=0.001, restore_best=True)

    tr_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train.reshape(-1, 1)).float(),
    )
    va_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val.reshape(-1, 1)).float(),
    )

    dl_gen = torch.Generator()
    dl_gen.manual_seed(fold_seed + 9)
    worker_init = make_dataloader_worker_init_fn(fold_seed + 99) if num_workers > 0 else None

    tr_dl = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        generator=dl_gen,
        worker_init_fn=worker_init,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        worker_init_fn=worker_init,
    )

    for _epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_loss += float(loss_fn(logits, yb).item())
        stopper(val_loss, model, _epoch)
        if stopper.early_stop:
            break

    stopper.restore(model)

    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        model.eval()
        for xb, _ in va_dl:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def _collect_oof_predictions(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, float]]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install train dependencies first: uv sync --extra train") from e

    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_json = args.dataset_json.expanduser().resolve()

    paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dataset_json))
    y = np.asarray(y_list, dtype=np.float64)
    y_int = y.astype(np.int64)
    indices = np.arange(len(paths))

    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_pairs = list(splitter.split(indices, y_int))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_pairs = list(splitter.split(indices))

    rows_by_tid = _load_rows_by_tender_id(dataset_json)
    tender_ids = _aligned_tender_ids(paths)
    texts = _load_cached_texts(
        tender_ids=tender_ids,
        rows_by_tid=rows_by_tid,
        cache_dir=args.text_cache_dir.expanduser().resolve(),
    )
    meanpooled = _load_meanpooled_embeddings(paths)
    base_ds = CachedChunkEmbDataset(files=paths, y_list=y_list)

    cfg = default_cfg()
    cfg.cache_dir = str(cache_dir)

    oof_probs = {
        "tfidf_logreg": np.full(len(paths), np.nan, dtype=np.float64),
        "meanpool_mlp": np.full(len(paths), np.nan, dtype=np.float64),
        "cross_chunk_transformer": np.full(len(paths), np.nan, dtype=np.float64),
    }
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(split_pairs):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)
        fold_seed = _fold_rng_seed(args.seed, fold)
        configure_reproducibility(fold_seed)

        x_train_text = [texts[i] for i in tr_idx]
        x_val_text = [texts[i] for i in va_idx]
        y_train = y[tr_idx]

        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5)
        x_train_tfidf = vec.fit_transform(x_train_text)
        x_val_tfidf = vec.transform(x_val_text)
        tfidf_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        tfidf_clf.fit(x_train_tfidf, y_train.astype(np.int64))
        oof_probs["tfidf_logreg"][va_idx] = tfidf_clf.predict_proba(x_val_tfidf)[:, 1]

        oof_probs["meanpool_mlp"][va_idx] = _train_eval_meanpool_mlp(
            x_train=meanpooled[tr_idx],
            y_train=y[tr_idx],
            x_val=meanpooled[va_idx],
            y_val=y[va_idx],
            d_in=meanpooled.shape[1],
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.mlp_dropout,
            batch_size=args.mlp_batch_size,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            patience=args.patience,
            fold_seed=fold_seed,
            num_workers=args.num_workers,
        )

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
        oof_probs["cross_chunk_transformer"][va_idx] = probs_t.numpy().reshape(-1)

        fold_metrics.append(
            {
                "fold": float(fold),
                "positive_prevalence": float(np.mean(y_t.numpy().reshape(-1))),
            }
        )

    return y, oof_probs, fold_metrics


def _plot_pr_curve(*, y: np.ndarray, probs_by_model: dict[str, np.ndarray], fig_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    labels = {
        "tfidf_logreg": "TF-IDF + LogReg",
        "meanpool_mlp": "MeanPool + MLP",
        "cross_chunk_transformer": "Cross-chunk Transformer",
    }
    colors = {
        "tfidf_logreg": "tab:blue",
        "meanpool_mlp": "tab:orange",
        "cross_chunk_transformer": "tab:green",
    }

    prevalence = float(np.mean(y))
    summary: dict[str, float] = {"positive_prevalence": prevalence}

    plt.figure(figsize=(5.2, 4.0))
    for model_key in ("tfidf_logreg", "meanpool_mlp", "cross_chunk_transformer"):
        precision, recall, _ = precision_recall_curve(y, probs_by_model[model_key])
        metrics = binary_classification_metrics(probs_by_model[model_key], y, threshold=0.5)
        summary[f"{model_key}_pr_auc_oof"] = float(metrics["pr_auc"])
        plt.plot(
            recall,
            precision,
            linewidth=2.0,
            color=colors[model_key],
            label=f"{labels[model_key]} (PR AUC={float(metrics['pr_auc']):.3f})",
        )

    plt.axhline(prevalence, color="tab:red", linestyle="--", linewidth=1.4, label=f"Prevalence ({prevalence:.3f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Comparative Precision-Recall Curves")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close()
    return summary


def _plot_calibration_curve(*, y: np.ndarray, probs_by_model: dict[str, np.ndarray], fig_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    labels = {
        "tfidf_logreg": "TF-IDF + LogReg",
        "meanpool_mlp": "MeanPool + MLP",
        "cross_chunk_transformer": "Cross-chunk Transformer",
    }
    colors = {
        "tfidf_logreg": "tab:blue",
        "meanpool_mlp": "tab:orange",
        "cross_chunk_transformer": "tab:green",
    }

    summary: dict[str, float] = {}
    plt.figure(figsize=(5.2, 4.0))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
    for model_key in ("tfidf_logreg", "meanpool_mlp", "cross_chunk_transformer"):
        frac_pos, mean_pred = calibration_curve(y, probs_by_model[model_key], n_bins=10, strategy="quantile")
        metrics = binary_classification_metrics(probs_by_model[model_key], y, threshold=0.5)
        summary[f"{model_key}_brier_oof"] = float(metrics["brier_score"])
        plt.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2.0,
            color=colors[model_key],
            label=f"{labels[model_key]} (Brier={float(metrics['brier_score']):.3f})",
        )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency of positives")
    plt.title("Comparative Calibration Curves")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close()
    return summary


def _plot_threshold_sensitivity(*, threshold_json_path: Path, fig_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    raw = json.loads(threshold_json_path.read_text(encoding="utf-8"))
    models = raw["models"]
    order = ["tfidf_logreg", "meanpool_logreg", "meanpool_mlp", "cross_chunk_transformer"]
    labels = ["TF-IDF + LogReg", "MeanPool + LogReg", "MeanPool + MLP", "Transformer"]
    delta_f1 = [float(models[key]["summary"]["delta_f1"]["mean"]) for key in order]
    delta_bal = [float(models[key]["summary"]["delta_balanced_accuracy"]["mean"]) for key in order]

    x = np.arange(len(order))
    width = 0.36

    plt.figure(figsize=(5.2, 3.7))
    plt.axhline(0.0, color="gray", linewidth=1.0)
    plt.bar(x - width / 2, delta_f1, width=width, color="tab:blue", label="Delta F1")
    plt.bar(x + width / 2, delta_bal, width=width, color="tab:orange", label="Delta balanced acc.")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Mean change after threshold tuning")
    plt.title("Threshold Tuning Sensitivity")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close()

    return {
        f"{key}_delta_f1_mean": value
        for key, value in zip(order, delta_f1, strict=False)
    } | {
        f"{key}_delta_balanced_accuracy_mean": value
        for key, value in zip(order, delta_bal, strict=False)
    }


def main() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install train dependencies first: uv sync --extra train") from e

    load_dotenv()
    args = _parse_args()
    configure_reproducibility(args.seed)
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    y, probs_by_model, fold_metrics = _collect_oof_predictions(args)

    fig_dir = PAPER_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PAPER_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_summary = _plot_pr_curve(y=y, probs_by_model=probs_by_model, fig_path=fig_dir / "pr_curve_cv.png")
    cal_summary = _plot_calibration_curve(
        y=y,
        probs_by_model=probs_by_model,
        fig_path=fig_dir / "calibration_curve_cv.png",
    )
    threshold_summary = _plot_threshold_sensitivity(
        threshold_json_path=args.threshold_comparison_json.expanduser().resolve(),
        fig_path=fig_dir / "threshold_sensitivity_comparison.png",
    )

    metrics_by_model = {
        model_key: {
            key: float(value)
            for key, value in binary_classification_metrics(
                probs,
                y,
                threshold=args.class_threshold,
            ).items()
            if isinstance(value, float)
        }
        for model_key, probs in probs_by_model.items()
    }

    out = {
        "dataset_json": str(args.dataset_json.expanduser().resolve()),
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "fold_metrics": fold_metrics,
        "oof_metrics": metrics_by_model,
        "pr_figure_summary": pr_summary,
        "calibration_figure_summary": cal_summary,
        "threshold_figure_summary": threshold_summary,
    }
    (out_dir / "comparative_paper_figures_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
