"""
K-fold cross-validation del predictor sobre embeddings `.pt` con `y`, con registro en MLflow.

Requiere dependencias de entrenamiento:
  uv sync --extra train

Artefactos (modelos, ficheros) en DO Spaces: …/outcome-predictor/mlflow (misma base que
procurements), vía credenciales SPACES_* en `.env`. Métricas/params: SQLite en data/mlflow.db
salvo que definas MLFLOW_TRACKING_URI.

Misma ``--seed`` y mismos datos → mismas divisiones CV y RNG acotada por fold (cuDNN
determinista y ``DataLoader`` con ``generator``). Para CUDA suele ayudar
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (el script lo define si falta).

Embeddings locales (si están solo en Spaces):
  uv run python scripts/sync_embeddings_for_training.py

Etiqueta desde status del JSON (complete=1, unsuccessful|cancelled=0), alineada con tender_id del .pt:
  uv run python scripts/train_cv_mlflow.py --cache-dir data/chunk_embeddings \\
    --dataset-json /ruta/a/procurements_dataset.json --folds 5

Ejemplos:
  uv run python scripts/train_cv_mlflow.py --cache-dir data/chunk_embeddings --folds 5

  MLFLOW_TRACKING_URI=sqlite:////abs/path/mlflow.db \\
    uv run python scripts/train_cv_mlflow.py ...

  Override URI de artefactos: --artifact-root s3://bucket/prefijo/mlflow
"""

from __future__ import annotations

import argparse
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
    list_labeled_embedding_paths,
    list_labeled_embedding_paths_from_dataset_json,
)
from models.predictor import build_model_from_sample_batch
from training.loop import evaluate_probs, train_one_fold
from training.metrics import binary_classification_metrics
from training.reproducibility import configure_reproducibility, make_dataloader_worker_init_fn
from training.mlflow_spaces import (
    configure_mlflow_s3_env_from_spaces,
    ensure_mlflow_experiment,
    spaces_mlflow_artifact_root,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV + MLflow para TenderSuccessPredictor.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
        help="Directorio con .pt por licitación (embed_pbcs / caché local).",
    )
    p.add_argument("--folds", type=int, default=5, help="Número de folds (K).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="StratifiedKFold si todas las y son 0/1; si no, KFold.",
    )
    p.add_argument("--experiment", type=str, default="procurements_predictor", help="Nombre MLflow experiment.")
    p.add_argument("--run-name", type=str, default=None, help="Nombre del run padre en MLflow.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=5, help="Early stopping (train_one_fold).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--class-threshold",
        type=float,
        default=0.5,
        help="Umbral en prob. para F1 y balanced accuracy (no afecta ROC/PR/Brier).",
    )
    p.add_argument("--log-models", action="store_true", help="Loggear un artefacto PyTorch por fold (nested run).")
    p.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="S3 URI raíz de artefactos MLflow (default: s3://SPACES_BUCKET/…/outcome-predictor/mlflow).",
    )
    p.add_argument(
        "--no-spaces-artifacts",
        action="store_true",
        help="No usar S3: experimento MLflow con ubicación por defecto del tracking store.",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help=(
            "procurements_dataset.json: etiquetas desde status (complete=1, unsuccessful|cancelled=0). "
            "Si se omite, se usa la clave y dentro de cada .pt."
        ),
    )
    return p.parse_args()


def _fold_rng_seed(global_seed: int, fold: int) -> int:
    """Semilla estable por fold (no depende del número de épocas del fold anterior)."""
    return int(global_seed) + int(fold) * 1_000_003


def _use_stratified(y: np.ndarray, stratify: bool) -> bool:
    if not stratify or len(np.unique(y)) < 2:
        return False
    return bool(np.all(np.isclose(y, 0) | np.isclose(y, 1)))


def main() -> None:
    try:
        import mlflow
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Falta mlflow. Ejecutá: uv sync --extra train",
        ) from e
    from sklearn.model_selection import KFold, StratifiedKFold

    load_dotenv()
    args = _parse_args()
    configure_reproducibility(args.seed)
    configure_mlflow_s3_env_from_spaces()

    if not os.environ.get("MLFLOW_TRACKING_URI", "").strip():
        db_path = (REPO_ROOT / "data" / "mlflow.db").resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path.as_posix()}"

    cache_dir = args.cache_dir.resolve()
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
        raise SystemExit(
            f"Hay {len(paths)} muestras etiquetadas; hacen falta al menos --folds={args.folds}.",
        )

    y = np.asarray(y_list, dtype=np.float64)
    y_strat = y.astype(np.int64)

    cfg = default_cfg()
    cfg.cache_dir = str(cache_dir)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr

    indices = np.arange(len(paths))
    if _use_stratified(y, args.stratify):
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        splits = splitter.split(indices, y_strat)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(indices)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    env_art = os.environ.get("MLFLOW_SPACES_ARTIFACT_ROOT", "").strip()
    if args.no_spaces_artifacts:
        artifact_root: str | None = None
    elif args.artifact_root:
        artifact_root = args.artifact_root.strip()
    elif env_art:
        artifact_root = env_art
    else:
        artifact_root = spaces_mlflow_artifact_root()

    if artifact_root:
        print(f"MLflow artifact_location: {artifact_root}")

    ensure_mlflow_experiment(mlflow, name=args.experiment, artifact_root=artifact_root)
    parent_name = args.run_name or f"cv_{args.folds}fold_seed{args.seed}"

    params_base = {
        "folds": args.folds,
        "seed": args.seed,
        "cache_dir": str(cache_dir),
        "n_labeled": len(paths),
        "stratified": _use_stratified(y, args.stratify),
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "device": cfg.device,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "ffn_dim": cfg.ffn_dim,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
        "early_stopping_patience": args.patience,
        "class_threshold": args.class_threshold,
        "dataset_json": str(args.dataset_json.resolve()) if args.dataset_json else None,
        "target_status_rule": "complete=1, unsuccessful|cancelled|canceled=0",
        "reproducibility": "seed+cudnn_deterministic+dataloader_generator+fold_resets",
    }

    fold_rows: list[dict[str, float]] = []

    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(params_base)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr_idx = np.asarray(tr_idx, dtype=np.int64)
            va_idx = np.asarray(va_idx, dtype=np.int64)

            fold_seed = _fold_rng_seed(args.seed, fold)
            configure_reproducibility(fold_seed)

            base_ds = CachedChunkEmbDataset(files=paths, y_list=y_list)
            train_ds = Subset(base_ds, tr_idx.tolist())
            val_ds = Subset(base_ds, va_idx.tolist())

            dl_gen = torch.Generator()
            dl_gen.manual_seed(fold_seed + 9)
            worker_init = (
                make_dataloader_worker_init_fn(fold_seed + 99)
                if args.num_workers > 0
                else None
            )

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

            sample_batch = cast(
                List[Tuple[torch.Tensor, torch.Tensor]],
                [train_ds[0]],
            )
            embs0, _, _ = collate_pad_chunks(sample_batch)
            model = build_model_from_sample_batch(embs0, cfg)

            with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):
                mlflow.log_param("fold", fold)
                mlflow.log_param("n_train", int(len(tr_idx)))
                mlflow.log_param("n_val", int(len(va_idx)))

                history, best_val_loss, best_epoch = train_one_fold(
                    model,
                    train_dl,
                    val_dl,
                    cfg,
                    fold=fold,
                    patience=args.patience,
                )
                mlflow.log_metric("best_val_loss", float(best_val_loss))
                mlflow.log_metric("best_epoch", int(best_epoch))

                probs_t, y_t = evaluate_probs(model, val_dl, cfg)
                probs = probs_t.numpy().reshape(-1)
                y_true = y_t.numpy().reshape(-1)
                mask = y_true >= 0
                metrics = binary_classification_metrics(
                    probs[mask],
                    y_true[mask],
                    threshold=args.class_threshold,
                )

                for name, value in metrics.items():
                    if isinstance(value, float) and value == value:
                        mlflow.log_metric(name, value)

                last_ep = len(history["train_loss"])
                mlflow.log_metric("epochs_ran", float(last_ep))

                if args.log_models:
                    import mlflow.pytorch as mlflow_pytorch

                    mlflow_pytorch.log_model(model, artifact_path="model")

                fold_rows.append(metrics)

        for key in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "brier_score", "log_loss"):
            vals = []
            for row in fold_rows:
                v = row.get(key)
                if isinstance(v, float) and v == v:
                    vals.append(v)
            if vals:
                mlflow.log_metric(f"cv_mean_{key}", float(np.mean(vals)))
                mlflow.log_metric(f"cv_std_{key}", float(np.std(vals)))


if __name__ == "__main__":
    main()
