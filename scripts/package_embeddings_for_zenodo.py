"""Build a validated, documented ZIP of chunk embeddings for Zenodo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {
    "chunk_batch_size",
    "embs",
    "max_len",
    "model_id",
    "stride",
    "tender_id",
    "y",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(path: Path) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a dictionary")
    missing = REQUIRED_KEYS.difference(value)
    if missing:
        raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
    if not isinstance(value["embs"], torch.Tensor) or value["embs"].ndim != 2:
        raise ValueError(f"{path.name}: embs must be a rank-2 tensor")
    return value


def readme(metadata: dict) -> str:
    return f"""# Public procurement chunk embeddings

This archive contains the frozen chunk embeddings used in the experiments for
“Lexical and LLM-Based Representations for Public Tender Outcome Prediction:
A Comparative Study”.

## Contents

- `embeddings/`: {metadata["file_count"]} PyTorch `.pt` files, one per procurement.
- `manifest.csv`: filename, SHA-256 checksum, size and tensor metadata.
- `metadata.json`: aggregate archive metadata.

Each `.pt` dictionary contains `tender_id`, `embs`, `model_id`, `max_len`,
`stride`, `chunk_batch_size` and `y`. The `embs` tensor stores one frozen
representation per document chunk. Labels use `1` for `complete` and `0` for
`unsuccessful`, `cancelled` or `canceled`.

Encoder: `{metadata["model_ids"][0]}`
Embedding dimension: {metadata["embedding_dimensions"][0]}
Chunk length: {metadata["max_lengths"][0]} tokens
Stride: {metadata["strides"][0]} tokens

These files reproduce the dense-input stage of the reported experiments. The
code and environment specification are archived separately in the associated
Zenodo software record.

PyTorch files use pickle-based serialization. Load only files obtained from the
official archive and verify their checksums against `manifest.csv`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chunk_embeddings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "zenodo"
        / "public-procurements-chunk-embeddings-v1.0.zip",
    )
    parser.add_argument("--expected-count", type=int, default=1162)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.pt"))
    if len(paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} files, found {len(paths)}")

    manifest_rows: list[dict[str, str | int | float]] = []
    labels: Counter[str] = Counter()
    model_ids: set[str] = set()
    dimensions: set[int] = set()
    max_lengths: set[int] = set()
    strides: set[int] = set()
    dtypes: set[str] = set()
    total_chunks = 0

    for path in paths:
        item = load_embedding(path)
        embs = item["embs"]
        label = str(int(float(item["y"])))
        labels[label] += 1
        model_ids.add(str(item["model_id"]))
        dimensions.add(int(embs.shape[1]))
        max_lengths.add(int(item["max_len"]))
        strides.add(int(item["stride"]))
        dtypes.add(str(embs.dtype))
        total_chunks += int(embs.shape[0])
        manifest_rows.append(
            {
                "filename": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "tender_id": str(item["tender_id"]),
                "label": label,
                "chunks": int(embs.shape[0]),
                "embedding_dimension": int(embs.shape[1]),
                "dtype": str(embs.dtype),
                "model_id": str(item["model_id"]),
                "max_len": int(item["max_len"]),
                "stride": int(item["stride"]),
                "chunk_batch_size": int(item["chunk_batch_size"]),
            }
        )

    metadata = {
        "file_count": len(paths),
        "total_chunks": total_chunks,
        "labels": dict(sorted(labels.items())),
        "model_ids": sorted(model_ids),
        "embedding_dimensions": sorted(dimensions),
        "max_lengths": sorted(max_lengths),
        "strides": sorted(strides),
        "dtypes": sorted(dtypes),
    }

    manifest_buffer = io.StringIO()
    writer = csv.DictWriter(manifest_buffer, fieldnames=list(manifest_rows[0]))
    writer.writeheader()
    writer.writerows(manifest_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.writestr("README.md", readme(metadata))
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr("manifest.csv", manifest_buffer.getvalue())
        for path in paths:
            archive.write(path, f"embeddings/{path.name}")

    print(args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"archive_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
