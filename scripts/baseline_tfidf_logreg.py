"""
Baseline text classification experiment with TF-IDF + Logistic Regression.

Uses the same supervised subset and the same CV split logic as `train_cv_mlflow.py`.
Raw text is taken before chunking, from the extracted PBC `.txt` object associated
with each procurement in the dataset JSON.

Example:
  HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \
    uv run python scripts/baseline_tfidf_logreg.py \
      --cache-dir data/chunk_embeddings \
      --dataset-json data/processed/procurements_dataset_s3_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
_ETL = REPO_ROOT / "scripts" / "etl"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ETL) not in sys.path:
    sys.path.insert(0, str(_ETL))

from data.chunk_dataset import list_labeled_embedding_paths_from_dataset_json
from training.mlflow_spaces import (
    configure_mlflow_s3_env_from_spaces,
    ensure_mlflow_experiment,
    spaces_mlflow_artifact_root,
)
from training.metrics import binary_classification_metrics
from training.reproducibility import configure_reproducibility

import spaces_io


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline TF-IDF + Logistic Regression with same CV splits.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
        help="Local directory with embedding .pt files; used only to align the supervised subset and folds.",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Dataset snapshot JSON used as source of labels and PBC txt keys.",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "pbc_txt_cache",
        help="Optional local cache for raw txt files fetched from Spaces.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "tfidf_logreg_baseline.json",
        help="Where to save the structured dict result.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="procurements_predictor",
        help="MLflow experiment name.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow parent run name.",
    )
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


def _use_stratified(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2 and bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


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


def _safe_name(tid: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in tid)[:220]


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


def main() -> None:
    try:
        import mlflow
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
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
    dataset_json = args.dataset_json.expanduser().resolve()
    rows_by_tid = _load_rows_by_tender_id(dataset_json)

    paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dataset_json))
    if len(paths) < args.folds:
        raise SystemExit(f"Need at least {args.folds} labeled samples, found {len(paths)}")

    tender_ids = _aligned_tender_ids(paths)
    texts = _load_or_fetch_texts(
        tender_ids=tender_ids,
        rows_by_tid=rows_by_tid,
        cache_dir=args.text_cache_dir.expanduser().resolve(),
    )

    y = np.asarray(y_list, dtype=np.float64)
    y_int = y.astype(np.int64)
    indices = np.arange(len(texts))

    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(indices, y_int)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(indices)

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
    parent_name = args.run_name or f"baseline_tfidf_logreg_cv_{args.folds}fold_seed{args.seed}"

    params_base = {
        "model": "tfidf_logreg",
        "folds": args.folds,
        "seed": args.seed,
        "cache_dir": str(cache_dir),
        "dataset_json": str(dataset_json),
        "text_cache_dir": str(args.text_cache_dir.expanduser().resolve()),
        "n_labeled": len(paths),
        "stratified": _use_stratified(y),
        "class_threshold": args.class_threshold,
        "tfidf_max_features": 50_000,
        "tfidf_ngram_range": "1,2",
        "tfidf_min_df": 5,
        "logreg_class_weight": "balanced",
        "logreg_max_iter": 1000,
        "target_status_rule": "complete=1, unsuccessful|cancelled|canceled=0",
        "reproducibility": "seed+deterministic_cv_split",
    }

    fold_metrics: list[dict[str, float]] = []
    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(params_base)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr_idx = np.asarray(tr_idx, dtype=np.int64)
            va_idx = np.asarray(va_idx, dtype=np.int64)

            x_train = [texts[i] for i in tr_idx]
            x_val = [texts[i] for i in va_idx]
            y_train = y[tr_idx]
            y_val = y[va_idx]

            vec = TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                min_df=5,
            )
            x_train_tfidf = vec.fit_transform(x_train)
            x_val_tfidf = vec.transform(x_val)

            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            )
            clf.fit(x_train_tfidf, y_train.astype(np.int64))

            probs = clf.predict_proba(x_val_tfidf)[:, 1]
            metrics = binary_classification_metrics(
                probs,
                y_val,
                threshold=args.class_threshold,
            )
            row = {
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "f1": float(metrics["f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "brier_score": float(metrics["brier_score"]),
            }
            fold_metrics.append(row)
            print(
                f"[Fold {fold}] "
                f"roc_auc={row['roc_auc']:.4f} "
                f"pr_auc={row['pr_auc']:.4f} "
                f"f1={row['f1']:.4f} "
                f"bal_acc={row['balanced_accuracy']:.4f} "
                f"brier={row['brier_score']:.4f}"
            )

            with mlflow.start_run(nested=True, run_name=f"baseline_fold_{fold}"):
                mlflow.log_param("fold", fold)
                mlflow.log_param("n_train", int(len(tr_idx)))
                mlflow.log_param("n_val", int(len(va_idx)))
                mlflow.log_param("vocab_size", int(x_train_tfidf.shape[1]))
                for key, value in row.items():
                    mlflow.log_metric(key, value)

        summary_metrics: dict[str, dict[str, float]] = {}
        for key in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "brier_score"):
            vals = np.asarray([row[key] for row in fold_metrics], dtype=np.float64)
            summary_metrics[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
            mlflow.log_metric(f"cv_mean_{key}", summary_metrics[key]["mean"])
            mlflow.log_metric(f"cv_std_{key}", summary_metrics[key]["std"])

        result = {
            "model": "tfidf_logreg",
            "metrics": summary_metrics,
        }

        args.out = args.out.expanduser().resolve()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(args.out), artifact_path="baseline_results")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
