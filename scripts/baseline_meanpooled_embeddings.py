"""
Embedding baselines over existing chunk embeddings.

For each procurement, chunk embeddings are aggregated with simple mean pooling
to obtain one fixed-size vector per document. Then one of the following models
is trained with the same CV split logic as the main cross-chunk transformer:

- LogisticRegression over mean-pooled embeddings
- Small MLP with one hidden layer over mean-pooled embeddings

If a transformer reference JSON is available, the script also emits a compact
comparison against the full cross-chunk transformer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.chunk_dataset import (
    list_labeled_embedding_paths,
    list_labeled_embedding_paths_from_dataset_json,
)
from training.early_stopping import EarlyStopping
from training.metrics import binary_classification_metrics
from training.mlflow_spaces import (
    configure_mlflow_s3_env_from_spaces,
    ensure_mlflow_experiment,
    spaces_mlflow_artifact_root,
)
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
    p = argparse.ArgumentParser(description="Mean-pooled embedding baselines with same CV splits.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
        help="Directory with .pt chunk embedding files.",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Optional dataset snapshot JSON to override labels from status, same as train_cv_mlflow.py.",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-type",
        type=str,
        choices=("logreg", "mlp", "both"),
        default="both",
        help="Which mean-pool baseline to run.",
    )
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--mlp-hidden-dim", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.2)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--mlp-epochs", type=int, default=50)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--transformer-reference-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_analysis_stats.json",
        help="Optional JSON with fold metrics for the full cross-chunk transformer.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "meanpooled_embedding_baselines.json",
        help="Where to save the structured dict result.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="procurements_predictor",
        help="MLflow experiment name.",
    )
    p.add_argument("--run-name", type=str, default=None, help="Optional MLflow parent run name.")
    p.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="MLflow artifact root override, e.g. s3://bucket/prefix/mlflow.",
    )
    p.add_argument(
        "--no-spaces-artifacts",
        action="store_true",
        help="Do not use the default Spaces-backed MLflow artifact location.",
    )
    return p.parse_args()


def _fold_rng_seed(global_seed: int, fold: int) -> int:
    return int(global_seed) + int(fold) * 1_000_003


def _use_stratified(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2 and bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


def _load_meanpooled_embeddings(paths: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        embs = d["embs"].float()
        pooled = embs.mean(dim=0)
        rows.append(pooled.numpy().astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in (
        "roc_auc",
        "pr_auc",
        "pr_auc_excess_over_prevalence",
        "positive_prevalence",
        "f1",
        "balanced_accuracy",
        "brier_score",
        "log_loss",
    ):
        vals = np.asarray(
            [row[key] for row in rows if isinstance(row.get(key), float) and row[key] == row[key]],
            dtype=np.float64,
        )
        if vals.size:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def _load_transformer_reference(path: Path) -> dict[str, object] | None:
    ref_path = path.expanduser().resolve()
    if not ref_path.is_file():
        return None
    raw = json.loads(ref_path.read_text(encoding="utf-8"))
    fold_metrics = raw.get("fold_metrics")
    if not isinstance(fold_metrics, list) or not fold_metrics:
        return None
    rows: list[dict[str, float]] = []
    for row in fold_metrics:
        if isinstance(row, dict):
            rows.append({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
    if not rows:
        return None
    return {
        "source": str(ref_path),
        "metrics": _summarize_metrics(rows),
    }


def _add_delta_vs_transformer(
    model_summary: dict[str, dict[str, float]],
    transformer_summary: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    if transformer_summary is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for key, stats in model_summary.items():
        ref = transformer_summary.get(key)
        if ref is None:
            continue
        out[key] = {"mean_delta": float(stats["mean"] - ref["mean"])}
    return out


def _train_eval_logreg(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )
    clf.fit(x_train, y_train.astype(np.int64))
    probs = clf.predict_proba(x_val)[:, 1]
    metrics = binary_classification_metrics(probs, y_val, threshold=threshold)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, float)}


def _train_eval_mlp(
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
    threshold: float,
    fold_seed: int,
    num_workers: int,
) -> dict[str, float]:
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

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_total += float(loss.item())

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_total += float(loss_fn(logits, yb).item())

        train_loss = train_total / max(1, len(tr_dl))
        val_loss = val_total / max(1, len(va_dl))
        print(f"[meanpool_mlp] epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        stopper(val_loss, model, epoch)
        if stopper.early_stop:
            print(f"[meanpool_mlp] early stopping at epoch {epoch}")
            break

    stopper.restore(model)

    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        model.eval()
        for xb, _ in va_dl:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs_list.append(probs)
    probs_np = np.concatenate(probs_list, axis=0)
    metrics = binary_classification_metrics(probs_np, y_val, threshold=threshold)
    out = {k: float(v) for k, v in metrics.items() if isinstance(v, float)}
    out["best_val_loss"] = float(stopper.best_loss)
    out["best_epoch"] = float(stopper.best_epoch)
    return out


def _run_model(
    *,
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    transformer_summary: dict[str, dict[str, float]] | None,
    mlflow: object,
) -> dict[str, object]:
    parent_name = args.run_name or f"meanpool_{model_type}_cv_{args.folds}fold_seed{args.seed}"
    params_base = {
        "model": f"meanpool_{model_type}",
        "pooling": "mean",
        "folds": args.folds,
        "seed": args.seed,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "dataset_json": str(args.dataset_json.expanduser().resolve()) if args.dataset_json else None,
        "n_labeled": int(len(y)),
        "stratified": _use_stratified(y),
        "class_threshold": args.class_threshold,
        "embedding_dim": int(x.shape[1]),
        "target_status_rule": "complete=1, unsuccessful|cancelled|canceled=0",
        "compare_against": "full_cross_chunk_transformer",
    }
    if model_type == "logreg":
        params_base.update(
            {
                "logreg_class_weight": "balanced",
                "logreg_max_iter": 1000,
            }
        )
    else:
        params_base.update(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "mlp_dropout": args.mlp_dropout,
                "mlp_batch_size": args.mlp_batch_size,
                "mlp_epochs": args.mlp_epochs,
                "mlp_lr": args.mlp_lr,
                "mlp_weight_decay": args.mlp_weight_decay,
                "early_stopping_patience": args.patience,
            }
        )

    fold_rows: list[dict[str, float]] = []
    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(params_base)
        for fold, (tr_idx, va_idx) in enumerate(split_pairs):
            tr_idx = np.asarray(tr_idx, dtype=np.int64)
            va_idx = np.asarray(va_idx, dtype=np.int64)
            x_train = x[tr_idx]
            y_train = y[tr_idx]
            x_val = x[va_idx]
            y_val = y[va_idx]
            fold_seed = _fold_rng_seed(args.seed, fold)

            with mlflow.start_run(nested=True, run_name=f"{model_type}_fold_{fold}"):
                mlflow.log_param("fold", fold)
                mlflow.log_param("n_train", int(len(tr_idx)))
                mlflow.log_param("n_val", int(len(va_idx)))

                if model_type == "logreg":
                    metrics = _train_eval_logreg(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        threshold=args.class_threshold,
                    )
                else:
                    metrics = _train_eval_mlp(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        d_in=x.shape[1],
                        hidden_dim=args.mlp_hidden_dim,
                        dropout=args.mlp_dropout,
                        batch_size=args.mlp_batch_size,
                        epochs=args.mlp_epochs,
                        lr=args.mlp_lr,
                        weight_decay=args.mlp_weight_decay,
                        patience=args.patience,
                        threshold=args.class_threshold,
                        fold_seed=fold_seed,
                        num_workers=args.num_workers,
                    )
                for name, value in metrics.items():
                    if value == value:
                        mlflow.log_metric(name, float(value))
                fold_rows.append(metrics)

        summary = _summarize_metrics(fold_rows)
        for key, stats in summary.items():
            mlflow.log_metric(f"cv_mean_{key}", float(stats["mean"]))
            mlflow.log_metric(f"cv_std_{key}", float(stats["std"]))

        delta_vs_transformer = _add_delta_vs_transformer(summary, transformer_summary)
        for key, delta in delta_vs_transformer.items():
            mlflow.log_metric(f"delta_vs_transformer_{key}", float(delta["mean_delta"]))

        return {
            "metrics": summary,
            "delta_vs_transformer": delta_vs_transformer,
        }


def main() -> None:
    try:
        import mlflow
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install train dependencies first: uv sync --extra train") from e

    load_dotenv()
    args = _parse_args()
    configure_reproducibility(args.seed)
    configure_mlflow_s3_env_from_spaces()

    if not os.environ.get("MLFLOW_TRACKING_URI", "").strip():
        db_path = (REPO_ROOT / "data" / "mlflow.db").resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"No existe el directorio de embeddings: {cache_dir}")

    if args.dataset_json is not None:
        dj = args.dataset_json.expanduser().resolve()
        if not dj.is_file():
            raise SystemExit(f"No existe el dataset JSON: {dj}")
        paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dj))
    else:
        paths, y_list = list_labeled_embedding_paths(str(cache_dir))
    if len(paths) < args.folds:
        raise SystemExit(f"Hay {len(paths)} muestras etiquetadas; hacen falta al menos --folds={args.folds}.")

    x = _load_meanpooled_embeddings(paths)
    y = np.asarray(y_list, dtype=np.float64)
    indices = np.arange(len(paths))
    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_pairs = list(splitter.split(indices, y.astype(np.int64)))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_pairs = list(splitter.split(indices))

    env_art = os.environ.get("MLFLOW_SPACES_ARTIFACT_ROOT", "").strip()
    if args.no_spaces_artifacts:
        artifact_root: str | None = None
    elif args.artifact_root:
        artifact_root = args.artifact_root.strip()
    elif env_art:
        artifact_root = env_art
    else:
        artifact_root = spaces_mlflow_artifact_root()
    ensure_mlflow_experiment(mlflow, name=args.experiment, artifact_root=artifact_root)

    transformer_ref = _load_transformer_reference(args.transformer_reference_json)
    transformer_summary = None
    if transformer_ref is not None:
        transformer_summary = transformer_ref.get("metrics")
        if not isinstance(transformer_summary, dict):
            transformer_summary = None

    model_types = ["logreg", "mlp"] if args.model_type == "both" else [args.model_type]
    result: dict[str, object] = {}
    if transformer_ref is not None:
        result["transformer_reference"] = transformer_ref

    for model_type in model_types:
        result[f"meanpool_{model_type}"] = _run_model(
            model_type=model_type,
            x=x,
            y=y,
            split_pairs=split_pairs,
            args=args,
            transformer_summary=transformer_summary,
            mlflow=mlflow,
        )

    args.out = args.out.expanduser().resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
