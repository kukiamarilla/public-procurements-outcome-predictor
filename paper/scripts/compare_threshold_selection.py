"""
Compare threshold=0.5 vs validation-optimized threshold across models.

Protocol:
- Outer CV split: defines the test fold (same logic as the main model).
- Inner validation split: carved from the outer-train split and used only to
  choose the threshold that maximizes F1.
- Test evaluation: report F1 and balanced accuracy on the outer test fold using
  both threshold=0.5 and the optimized threshold.

Models:
- tfidf_logreg
- meanpool_logreg
- meanpool_mlp
- cross_chunk_transformer
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
_ETL = REPO_ROOT / "scripts" / "etl"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ETL) not in sys.path:
    sys.path.insert(0, str(_ETL))

from config import default_cfg
from data.chunk_dataset import (
    CachedChunkEmbDataset,
    collate_pad_chunks,
    list_labeled_embedding_paths,
    list_labeled_embedding_paths_from_dataset_json,
)
from models.predictor import build_model_from_sample_batch
from training.early_stopping import EarlyStopping
from training.loop import evaluate_probs, train_one_fold
from training.mlflow_spaces import (
    configure_mlflow_s3_env_from_spaces,
    ensure_mlflow_experiment,
    spaces_mlflow_artifact_root,
)
from training.reproducibility import configure_reproducibility, make_dataloader_worker_init_fn

import spaces_io


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
    p = argparse.ArgumentParser(description="Compare threshold 0.5 vs validation-optimized threshold.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-val-ratio", type=float, default=0.2)
    p.add_argument(
        "--models",
        nargs="+",
        choices=("tfidf_logreg", "meanpool_logreg", "meanpool_mlp", "cross_chunk_transformer"),
        default=("tfidf_logreg", "meanpool_logreg", "meanpool_mlp", "cross_chunk_transformer"),
    )
    p.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "pbc_txt_cache",
    )
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--transformer-batch-size", type=int, default=None)
    p.add_argument("--transformer-epochs", type=int, default=None)
    p.add_argument("--transformer-lr", type=float, default=None)
    p.add_argument("--transformer-patience", type=int, default=5)
    p.add_argument("--mlp-hidden-dim", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.2)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--mlp-epochs", type=int, default=50)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--mlp-patience", type=int, default=5)
    p.add_argument(
        "--out",
        type=Path,
        default=PAPER_ROOT / "outputs" / "threshold_selection_comparison.json",
    )
    p.add_argument("--experiment", type=str, default="procurements_predictor")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--artifact-root", type=str, default=None)
    p.add_argument("--no-spaces-artifacts", action="store_true")
    return p.parse_args()


def _fold_rng_seed(global_seed: int, fold: int) -> int:
    return int(global_seed) + int(fold) * 1_000_003


def _use_stratified(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2 and bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


def _safe_name(tid: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in tid)[:220]


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
    raw = dataset_json_path.read_text(encoding="utf-8")
    rows = json.loads(raw)
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


def _load_or_fetch_texts(
    *,
    tender_ids: list[str],
    rows_by_tid: dict[str, dict],
    cache_dir: Path,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = spaces_io.s3_client()
    bucket = spaces_io.bucket_name()
    texts: list[str] = []

    for tid in tender_ids:
        row = rows_by_tid.get(tid)
        if row is None:
            raise KeyError(f"TenderId not found in dataset-json: {tid}")
        txt_key = str(row.get("pbc_txt_s3_key") or "").strip()
        if not txt_key:
            raise ValueError(f"Missing pbc_txt_s3_key for tenderId={tid}")

        local_path = cache_dir / f"{_safe_name(tid)}.txt"
        if local_path.is_file():
            text = local_path.read_text(encoding="utf-8", errors="replace")
        else:
            raw = spaces_io.get_object_bytes(client, bucket, txt_key)
            text = raw.decode("utf-8", errors="replace")
            local_path.write_text(text, encoding="utf-8")
        texts.append(text)
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


def _split_outer_train_for_val(
    *,
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    y_train = y[outer_train_idx].astype(np.int64)
    stratify = y_train if _use_stratified(y_train.astype(np.float64)) else None
    tr_rel, va_rel = train_test_split(
        np.arange(len(outer_train_idx)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return outer_train_idx[tr_rel], outer_train_idx[va_rel]


def _threshold_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_hat = (probs >= threshold).astype(np.int64)
    y_bin = y_true.astype(np.int64)
    return {
        "f1": float(f1_score(y_bin, y_hat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, y_hat)),
    }


def _best_f1_threshold(probs: np.ndarray, y_true: np.ndarray, default_threshold: float) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([probs.reshape(-1), np.array([0.0, default_threshold, 1.0])]))
    best_threshold = float(default_threshold)
    best_f1 = -1.0

    for threshold in candidates.tolist():
        metrics = _threshold_metrics(probs, y_true, float(threshold))
        f1 = metrics["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = float(threshold)
            continue
        if abs(f1 - best_f1) <= 1e-12:
            if abs(float(threshold) - default_threshold) < abs(best_threshold - default_threshold):
                best_threshold = float(threshold)
            elif (
                abs(float(threshold) - default_threshold) == abs(best_threshold - default_threshold)
                and float(threshold) < best_threshold
            ):
                best_threshold = float(threshold)

    return best_threshold, best_f1


def _summarize_fold_comparison(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    def _stats(key: str) -> dict[str, float]:
        vals = np.asarray([row[key] for row in per_fold], dtype=np.float64)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "threshold_0_5_f1": _stats("test_f1_at_0_5"),
        "threshold_opt_f1": _stats("test_f1_at_opt"),
        "delta_f1": _stats("delta_f1"),
        "threshold_0_5_balanced_accuracy": _stats("test_balanced_accuracy_at_0_5"),
        "threshold_opt_balanced_accuracy": _stats("test_balanced_accuracy_at_opt"),
        "delta_balanced_accuracy": _stats("delta_balanced_accuracy"),
        "optimal_threshold": _stats("optimal_threshold"),
    }


def _train_predict_tfidf_logreg(
    *,
    texts: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    x_train = [texts[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    x_test = [texts[i] for i in test_idx]

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5)
    x_train_tfidf = vec.fit_transform(x_train)
    x_val_tfidf = vec.transform(x_val)
    x_test_tfidf = vec.transform(x_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(x_train_tfidf, y[train_idx].astype(np.int64))
    return clf.predict_proba(x_val_tfidf)[:, 1], clf.predict_proba(x_test_tfidf)[:, 1]


def _train_predict_meanpool_logreg(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(x[train_idx], y[train_idx].astype(np.int64))
    return clf.predict_proba(x[val_idx])[:, 1], clf.predict_proba(x[test_idx])[:, 1]


def _train_predict_meanpool_mlp(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    fold_seed: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_reproducibility(fold_seed)
    d_in = int(x.shape[1])
    model = MeanPoolMLP(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=patience, min_delta=0.001, restore_best=True)

    tr_ds = TensorDataset(
        torch.from_numpy(x[train_idx]).float(),
        torch.from_numpy(y[train_idx].reshape(-1, 1)).float(),
    )
    va_ds = TensorDataset(
        torch.from_numpy(x[val_idx]).float(),
        torch.from_numpy(y[val_idx].reshape(-1, 1)).float(),
    )
    te_ds = TensorDataset(
        torch.from_numpy(x[test_idx]).float(),
        torch.from_numpy(y[test_idx].reshape(-1, 1)).float(),
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
    te_dl = DataLoader(
        te_ds,
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
            loss = loss_fn(model(xb), yb)
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
                val_total += float(loss_fn(model(xb), yb).item())

        train_loss = train_total / max(1, len(tr_dl))
        val_loss = val_total / max(1, len(va_dl))
        print(f"[meanpool_mlp] epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        stopper(val_loss, model, epoch)
        if stopper.early_stop:
            print(f"[meanpool_mlp] early stopping at epoch {epoch}")
            break

    stopper.restore(model)

    def _predict(dl: DataLoader) -> np.ndarray:
        probs_list: list[np.ndarray] = []
        with torch.no_grad():
            model.eval()
            for xb, _ in dl:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)

    return _predict(va_dl), _predict(te_dl)


def _train_predict_transformer(
    *,
    paths: list[str],
    y_list: list[float],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = default_cfg()
    cfg.cache_dir = str(args.cache_dir.expanduser().resolve())
    if args.transformer_batch_size is not None:
        cfg.batch_size = args.transformer_batch_size
    if args.transformer_epochs is not None:
        cfg.epochs = args.transformer_epochs
    if args.transformer_lr is not None:
        cfg.lr = args.transformer_lr

    configure_reproducibility(fold_seed)
    base_ds = CachedChunkEmbDataset(files=paths, y_list=y_list)
    train_ds = Subset(base_ds, train_idx.tolist())
    val_ds = Subset(base_ds, val_idx.tolist())
    test_ds = Subset(base_ds, test_idx.tolist())

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
    test_dl = DataLoader(
        test_ds,
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
    train_one_fold(
        model,
        train_dl,
        val_dl,
        cfg,
        fold=int((fold_seed - args.seed) / 1_000_003),
        patience=args.transformer_patience,
    )

    val_probs, _ = evaluate_probs(model, val_dl, cfg)
    test_probs, _ = evaluate_probs(model, test_dl, cfg)
    return (
        val_probs.numpy().reshape(-1),
        test_probs.numpy().reshape(-1),
    )


def _evaluate_model(
    *,
    model_name: str,
    y: np.ndarray,
    paths: list[str],
    y_list: list[float],
    texts: list[str] | None,
    meanpooled_x: np.ndarray | None,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, object]:
    per_fold: list[dict[str, float]] = []
    for fold, (outer_train_idx, test_idx) in enumerate(outer_splits):
        outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        fold_seed = _fold_rng_seed(args.seed, fold)
        train_idx, val_idx = _split_outer_train_for_val(
            outer_train_idx=outer_train_idx,
            y=y,
            val_ratio=args.inner_val_ratio,
            seed=fold_seed,
        )

        if model_name == "tfidf_logreg":
            if texts is None:
                raise ValueError("texts are required for tfidf_logreg")
            val_probs, test_probs = _train_predict_tfidf_logreg(
                texts=texts,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        elif model_name == "meanpool_logreg":
            if meanpooled_x is None:
                raise ValueError("meanpooled_x is required for meanpool_logreg")
            val_probs, test_probs = _train_predict_meanpool_logreg(
                x=meanpooled_x,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        elif model_name == "meanpool_mlp":
            if meanpooled_x is None:
                raise ValueError("meanpooled_x is required for meanpool_mlp")
            val_probs, test_probs = _train_predict_meanpool_mlp(
                x=meanpooled_x,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                hidden_dim=args.mlp_hidden_dim,
                dropout=args.mlp_dropout,
                batch_size=args.mlp_batch_size,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                weight_decay=args.mlp_weight_decay,
                patience=args.mlp_patience,
                fold_seed=fold_seed,
                num_workers=args.num_workers,
            )
        elif model_name == "cross_chunk_transformer":
            val_probs, test_probs = _train_predict_transformer(
                paths=paths,
                y_list=y_list,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold_seed=fold_seed,
                args=args,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_val = y[val_idx]
        y_test = y[test_idx]
        optimal_threshold, val_best_f1 = _best_f1_threshold(val_probs, y_val, args.class_threshold)
        test_05 = _threshold_metrics(test_probs, y_test, args.class_threshold)
        test_opt = _threshold_metrics(test_probs, y_test, optimal_threshold)
        row = {
            "fold": float(fold),
            "optimal_threshold": float(optimal_threshold),
            "val_best_f1": float(val_best_f1),
            "test_f1_at_0_5": float(test_05["f1"]),
            "test_f1_at_opt": float(test_opt["f1"]),
            "delta_f1": float(test_opt["f1"] - test_05["f1"]),
            "test_balanced_accuracy_at_0_5": float(test_05["balanced_accuracy"]),
            "test_balanced_accuracy_at_opt": float(test_opt["balanced_accuracy"]),
            "delta_balanced_accuracy": float(test_opt["balanced_accuracy"] - test_05["balanced_accuracy"]),
        }
        per_fold.append(row)
        print(
            f"[{model_name}][fold {fold}] "
            f"threshold_opt={row['optimal_threshold']:.4f} "
            f"delta_f1={row['delta_f1']:.4f} "
            f"delta_bal_acc={row['delta_balanced_accuracy']:.4f}"
        )

    return {
        "per_fold": per_fold,
        "summary": _summarize_fold_comparison(per_fold),
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
        dataset_json = args.dataset_json.expanduser().resolve()
        if not dataset_json.is_file():
            raise SystemExit(f"No existe el dataset JSON: {dataset_json}")
        paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dataset_json))
    else:
        dataset_json = None
        paths, y_list = list_labeled_embedding_paths(str(cache_dir))

    if len(paths) < args.folds:
        raise SystemExit(f"Hay {len(paths)} muestras etiquetadas; hacen falta al menos --folds={args.folds}.")

    y = np.asarray(y_list, dtype=np.float64)
    indices = np.arange(len(paths))
    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        outer_splits = list(splitter.split(indices, y.astype(np.int64)))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        outer_splits = list(splitter.split(indices))

    texts: list[str] | None = None
    if "tfidf_logreg" in args.models:
        if dataset_json is None:
            raise SystemExit("--dataset-json is required for tfidf_logreg threshold comparison")
        rows_by_tid = _load_rows_by_tender_id(dataset_json)
        tender_ids = _aligned_tender_ids(paths)
        texts = _load_or_fetch_texts(
            tender_ids=tender_ids,
            rows_by_tid=rows_by_tid,
            cache_dir=args.text_cache_dir.expanduser().resolve(),
        )

    meanpooled_x: np.ndarray | None = None
    if "meanpool_logreg" in args.models or "meanpool_mlp" in args.models:
        meanpooled_x = _load_meanpooled_embeddings(paths)

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

    parent_name = args.run_name or f"threshold_compare_cv_{args.folds}fold_seed{args.seed}"
    result: dict[str, object] = {
        "protocol": {
            "outer_folds": args.folds,
            "inner_val_ratio": args.inner_val_ratio,
            "threshold_selection_objective": "maximize_f1_on_validation",
            "comparison": "threshold_0.5_vs_validation_optimal_threshold",
            "seed": args.seed,
        },
        "models": {},
    }

    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(
            {
                "folds": args.folds,
                "seed": args.seed,
                "inner_val_ratio": args.inner_val_ratio,
                "class_threshold_default": args.class_threshold,
                "models": ",".join(args.models),
                "dataset_json": str(dataset_json) if dataset_json else None,
            }
        )

        for model_name in args.models:
            model_result = _evaluate_model(
                model_name=model_name,
                y=y,
                paths=paths,
                y_list=y_list,
                texts=texts,
                meanpooled_x=meanpooled_x,
                outer_splits=outer_splits,
                args=args,
            )
            result["models"][model_name] = model_result
            summary = model_result["summary"]
            if isinstance(summary, dict):
                for metric_name, stats in summary.items():
                    if isinstance(stats, dict) and "mean" in stats and "std" in stats:
                        mlflow.log_metric(f"{model_name}_{metric_name}_mean", float(stats["mean"]))
                        mlflow.log_metric(f"{model_name}_{metric_name}_std", float(stats["std"]))

        args.out = args.out.expanduser().resolve()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(args.out), artifact_path="threshold_selection")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
