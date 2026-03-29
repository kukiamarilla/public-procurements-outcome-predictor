"""
Trivial classification baselines using the same supervised subset and CV splits as
the main model:

- majority_class: always predicts the majority class seen in the training fold
- random_prevalence: random predictions with positive rate equal to the training
  fold prevalence

Expected behavior for uninformed baselines on binary classification:
- ROC AUC ~ 0.5
- PR AUC ~ positive prevalence

Example:
  uv run python scripts/baseline_dummy_classifiers.py \
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
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.chunk_dataset import list_labeled_embedding_paths_from_dataset_json
from training.metrics import binary_classification_metrics
from training.mlflow_spaces import (
    configure_mlflow_s3_env_from_spaces,
    ensure_mlflow_experiment,
    spaces_mlflow_artifact_root,
)
from training.reproducibility import configure_reproducibility


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dummy classifier baselines with same CV splits.")
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
        help="Dataset snapshot JSON used as source of labels.",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-threshold", type=float, default=0.5)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "dummy_classifier_baselines.json",
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


def _fold_seed(global_seed: int, fold: int, offset: int) -> int:
    return int(global_seed) + int(fold) * 1_000_003 + int(offset)


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in ("roc_auc", "pr_auc", "f1", "balanced_accuracy"):
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return summary


def _print_fold_metrics(name: str, fold: int, row: dict[str, float]) -> None:
    print(
        f"[{name}][Fold {fold}] "
        f"roc_auc={row['roc_auc']:.4f} "
        f"pr_auc={row['pr_auc']:.4f} "
        f"f1={row['f1']:.4f} "
        f"bal_acc={row['balanced_accuracy']:.4f}"
    )


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
    dataset_json = args.dataset_json.expanduser().resolve()
    paths, y_list = list_labeled_embedding_paths_from_dataset_json(str(cache_dir), str(dataset_json))
    if len(paths) < args.folds:
        raise SystemExit(f"Need at least {args.folds} labeled samples, found {len(paths)}")

    y = np.asarray(y_list, dtype=np.float64)
    y_int = y.astype(np.int64)
    indices = np.arange(len(paths))

    if _use_stratified(y):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_pairs = list(splitter.split(indices, y_int))
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
    parent_name = args.run_name or f"baseline_dummy_random_cv_{args.folds}fold_seed{args.seed}"

    params_base = {
        "models": "majority_class,random_prevalence",
        "folds": args.folds,
        "seed": args.seed,
        "cache_dir": str(cache_dir),
        "dataset_json": str(dataset_json),
        "n_labeled": len(paths),
        "stratified": _use_stratified(y),
        "class_threshold": args.class_threshold,
        "target_status_rule": "complete=1, unsuccessful|cancelled|canceled=0",
        "expected_roc_auc": "~0.5",
        "expected_pr_auc": "~prevalence",
    }

    majority_rows: list[dict[str, float]] = []
    random_rows: list[dict[str, float]] = []

    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(params_base)

        for fold, (tr_idx, va_idx) in enumerate(split_pairs):
            tr_idx = np.asarray(tr_idx, dtype=np.int64)
            va_idx = np.asarray(va_idx, dtype=np.int64)

            y_train = y[tr_idx]
            y_val = y[va_idx]
            train_prevalence = float(np.mean(y_train))
            majority_positive = train_prevalence >= 0.5

            majority_probs = np.full(
                shape=y_val.shape,
                fill_value=1.0 if majority_positive else 0.0,
                dtype=np.float64,
            )
            majority_metrics = binary_classification_metrics(
                majority_probs,
                y_val,
                threshold=args.class_threshold,
            )
            majority_row = {
                "roc_auc": float(majority_metrics["roc_auc"]),
                "pr_auc": float(majority_metrics["pr_auc"]),
                "f1": float(majority_metrics["f1"]),
                "balanced_accuracy": float(majority_metrics["balanced_accuracy"]),
            }
            majority_rows.append(majority_row)
            _print_fold_metrics("majority_class", fold, majority_row)

            rng = np.random.default_rng(_fold_seed(args.seed, fold, offset=17))
            random_probs = rng.binomial(1, train_prevalence, size=len(y_val)).astype(np.float64)
            random_metrics = binary_classification_metrics(
                random_probs,
                y_val,
                threshold=args.class_threshold,
            )
            random_row = {
                "roc_auc": float(random_metrics["roc_auc"]),
                "pr_auc": float(random_metrics["pr_auc"]),
                "f1": float(random_metrics["f1"]),
                "balanced_accuracy": float(random_metrics["balanced_accuracy"]),
            }
            random_rows.append(random_row)
            _print_fold_metrics("random_prevalence", fold, random_row)

            with mlflow.start_run(nested=True, run_name=f"dummy_fold_{fold}"):
                mlflow.log_param("fold", fold)
                mlflow.log_param("n_train", int(len(tr_idx)))
                mlflow.log_param("n_val", int(len(va_idx)))
                mlflow.log_metric("train_positive_prevalence", train_prevalence)
                for key, value in majority_row.items():
                    mlflow.log_metric(f"majority_{key}", value)
                for key, value in random_row.items():
                    mlflow.log_metric(f"random_{key}", value)

        result = {
            "majority_class": {
                "metrics": _summarize_rows(majority_rows),
            },
            "random_prevalence": {
                "metrics": _summarize_rows(random_rows),
            },
            "expected_values": {
                "roc_auc": "~0.5 for uninformed ranking",
                "pr_auc": "~positive prevalence for random ranking in imbalanced data",
            },
        }

        for prefix, rows in (
            ("majority", majority_rows),
            ("random", random_rows),
        ):
            summary = _summarize_rows(rows)
            for metric_name, stats in summary.items():
                mlflow.log_metric(f"{prefix}_cv_mean_{metric_name}", stats["mean"])
                mlflow.log_metric(f"{prefix}_cv_std_{metric_name}", stats["std"])

        args.out = args.out.expanduser().resolve()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(args.out), artifact_path="baseline_results")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
