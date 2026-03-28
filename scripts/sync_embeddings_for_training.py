"""
Descarga desde DO Spaces los .pt bajo `pbcs/embeddings/` para entrenar con train_cv_mlflow.py.

Por defecto solo guarda archivos que incluyan la clave `y` (los que usa el CV).

  uv run python scripts/sync_embeddings_for_training.py
  uv run python scripts/sync_embeddings_for_training.py --out-dir data/chunk_embeddings --limit 500

Requiere .env con SPACES_* (igual que el resto de ETL).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_ETL = REPO_ROOT / "scripts" / "etl"
if str(_ETL) not in sys.path:
    sys.path.insert(0, str(_ETL))

import spaces_io  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Sync embedding .pt desde Spaces para entrenamiento.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
        help="Directorio local (mismo que --cache-dir en train_cv_mlflow).",
    )
    p.add_argument("--limit", type=int, default=0, help="Máximo de .pt a bajar (0 = sin límite).")
    p.add_argument(
        "--all-pts",
        action="store_true",
        help="Bajar todos los .pt aunque no tengan `y` (ocupa más disco; el CV ignora los sin etiqueta).",
    )
    args = p.parse_args()

    client = spaces_io.s3_client()
    bucket = spaces_io.bucket_name()
    prefix = spaces_io.pbc_embeddings_prefix_key()
    keys = sorted(
        k for k in spaces_io.list_object_keys_under_prefix(client, bucket, prefix) if k.endswith(".pt")
    )
    if args.limit > 0:
        keys = keys[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0
    for key in keys:
        raw = spaces_io.get_object_bytes(client, bucket, key)
        if not args.all_pts:
            try:
                d = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            except TypeError:
                d = torch.load(io.BytesIO(raw), map_location="cpu")
            if d.get("y") is None:
                n_skip += 1
                continue
        dest = args.out_dir / Path(key).name
        dest.write_bytes(raw)
        n_ok += 1
        print(dest)

    print(f"Listo: {n_ok} archivos en {args.out_dir.resolve()}", file=sys.stderr)
    if not args.all_pts:
        print(f"Omitidos sin `y`: {n_skip}", file=sys.stderr)


if __name__ == "__main__":
    main()
