"""
Genera embeddings por chunk (ChunkEmbedder) desde el texto PBC en Spaces y sube un .pt por licitación.

- Objetos: …/outcome-predictor/pbcs/embeddings/{stem}.pt
- Actualiza procurements_dataset.json con pbc_embedding_* y reconciliación contra pbcs/embeddings/.
- Checkpoint: reanuda si el .pt ya existe (salvo --force); persiste JSON cada --checkpoint-every filas.
- Una GPU por proceso: para varias GPUs, usá CUDA_VISIBLE_DEVICES + --shard i/n.

  uv run python scripts/etl/embed_pbcs.py
  uv run python scripts/etl/embed_pbcs.py --limit 50 --dry-run
  uv run python scripts/etl/embed_pbcs.py --chunk-batch-size 8 --checkpoint-every 25
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/etl/embed_pbcs.py --shard 1 4

  Sin truncar por defecto: chunks en streaming CPU→GPU; opcional --max-doc-tokens N.
  Fragmentación: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from dotenv import load_dotenv

import spaces_io
from models.embedder import build_chunk_embedder, forward_text_resolving_cuda_oom
from models.lm_config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "procurements_dataset.json"
DEFAULT_S3_DATASET_NAME = "procurements_dataset.json"


def _safe_file_stem(tender_id: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(tender_id).strip(), flags=re.ASCII)
    return (s[:max_len] if s else "unknown").strip("_") or "unknown"


def _training_y_from_row(row: dict) -> float | None:
    """Etiqueta en [0,1] si se puede inferir del dataset; si no, None."""
    for k in ("training_y", "outcome_y", "label"):
        v = row.get(k)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(max(0.0, min(1.0, float(v))))
        if isinstance(v, str) and v.strip():
            try:
                return float(max(0.0, min(1.0, float(v))))
            except ValueError:
                pass
    st = str(row.get("tenderStatus") or row.get("status") or "").strip().lower()
    if st in ("complete", "completo", "successful", "awarded", "adjudicado"):
        return 1.0
    if st in ("unsuccessful", "cancelled", "canceled", "desierto", "cancelado"):
        return 0.0
    return None


def _row_in_shard(tender_id: str, shard_index: int, shard_total: int) -> bool:
    if shard_total <= 1:
        return True
    h = zlib.adler32(tender_id.encode("utf-8"))
    return (h % shard_total) == shard_index


def _load_dataset(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _load_dataset_local_or_s3(
    path: Path,
    client: object,
    bucket: str,
    dataset_key: str,
) -> list[dict]:
    if path.is_file():
        return _load_dataset(path)
    print(f"Dataset local no encontrado; leyendo s3://{bucket}/{dataset_key}", file=sys.stderr)
    raw = spaces_io.get_object_bytes(client, bucket, dataset_key)
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, list) else []


def _apply_embedding_limit_defaults(
    merged: list[dict],
    candidate_ids: set[int],
    limit: int,
) -> None:
    if not limit or limit <= 0:
        return
    for row in merged:
        if not isinstance(row, dict):
            continue
        if row.get("pbc_text_extracted") is not True:
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        if id(row) in candidate_ids:
            continue
        row.setdefault("pbc_embedding_extracted", False)
        row.setdefault("pbc_embedding_skip_reason", "excluded_by_limit")
        row.setdefault(
            "pbc_embedding_error",
            "Excluida por --limit en embed_pbcs (no procesada en esta corrida).",
        )


def _reconcile_pbc_embedding_rows_with_spaces_prefix(
    merged: list[dict],
    client: object,
    bucket: str,
) -> None:
    prefix = spaces_io.pbc_embeddings_prefix_key()
    keys = frozenset(
        k for k in spaces_io.list_object_keys_under_prefix(client, bucket, prefix) if k.endswith(".pt")
    )
    print(
        f"Reconciliación embeddings PBC (fuente de verdad): "
        f"{len(keys)} .pt bajo s3://{bucket}/{prefix}",
    )
    for row in merged:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        stem_key = spaces_io.pbc_embedding_object_key(_safe_file_stem(tid))
        stored = str(row.get("pbc_embedding_s3_key") or "").strip()
        if stored and stored in keys:
            emb_key = stored
        elif stem_key in keys:
            emb_key = stem_key
        else:
            emb_key = None

        if emb_key is not None:
            row["pbc_embedding_extracted"] = True
            row["pbc_embedding_s3_key"] = emb_key
            row.pop("pbc_embedding_skip_reason", None)
            row.pop("pbc_embedding_error", None)
        else:
            was_ok = row.get("pbc_embedding_extracted") is True
            row["pbc_embedding_extracted"] = False
            row.pop("pbc_embedding_s3_key", None)
            if was_ok:
                row["pbc_embedding_skip_reason"] = "no_embedding_in_spaces"
                row["pbc_embedding_error"] = (
                    "El JSON indicaba embedding pero el .pt no está en el prefijo pbcs/embeddings/."
                )


def _save_embedding_pt(
    *,
    embs: torch.Tensor,
    tender_id: str,
    model_id: str,
    max_len: int,
    stride: int,
    chunk_batch_size: int,
    max_doc_tokens: int | None,
    y_optional: float | None,
) -> bytes:
    payload: dict[str, Any] = {
        "tender_id": tender_id,
        "embs": embs.cpu(),
        "model_id": model_id,
        "max_len": max_len,
        "stride": stride,
        "chunk_batch_size": chunk_batch_size,
        "max_doc_tokens": max_doc_tokens,
    }
    if y_optional is not None:
        payload["y"] = float(y_optional)
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _write_dataset_partial(
    merged: list[dict],
    *,
    local_out: Path,
    client: object,
    bucket: str,
    dataset_prefix: str,
    s3_name: str,
    no_upload: bool,
    dry_run: bool,
    candidate_ids: set[int],
    limit: int,
    do_reconcile: bool,
) -> None:
    _apply_embedding_limit_defaults(merged, candidate_ids, limit)
    if dry_run:
        return
    if do_reconcile:
        _reconcile_pbc_embedding_rows_with_spaces_prefix(merged, client, bucket)
    local_out.parent.mkdir(parents=True, exist_ok=True)
    with local_out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    if not no_upload:
        out_key = spaces_io.object_key(dataset_prefix, s3_name)
        spaces_io.put_json(client, bucket, out_key, merged)


def _print_row_done(
    done: int,
    total: int,
    tid: str,
    *,
    skipped: bool,
    work_seconds: float,
    work_done: int,
) -> None:
    if skipped:
        rate_s = f"omitidas ~{done / work_seconds * 60.0:.1f} filas/min" if work_seconds > 0 else ""
        print(f"  [{done}/{total}] tenderId={tid[:48]}…  (ya en Spaces)  {rate_s}".rstrip())
        return
    if work_seconds > 0 and work_done > 0:
        rate = work_done / work_seconds * 60.0
        rem = total - done
        eta = (rem / rate) if rate > 0 else 0.0
        print(
            f"  [{done}/{total}] tenderId={tid[:48]}…  "
            f"~{rate:.2f} docs/min · ETA ~{eta:.0f} min",
        )
    else:
        print(f"  [{done}/{total}] tenderId={tid[:48]}…")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Texto PBC (Spaces) → embeddings .pt en pbcs/embeddings/ + actualiza JSON",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--local-out", type=Path, default=None)
    parser.add_argument("--s3-output", default=DEFAULT_S3_DATASET_NAME)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = todas las filas elegibles")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recalcular aunque exista el .pt")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        metavar="N",
        help="Escribir procurements_dataset.json local (+ S3) cada N licitaciones procesadas (0 = solo al final)",
    )
    parser.add_argument("--model-id", default=None, help="Sobrescribe ModelConfig.model_id")
    parser.add_argument("--device", default=None, help='p. ej. "cuda", "cuda:0"')
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Opcional: truncar a N tokens (por defecto no se trunca). "
            "0 = sin truncar explícito (igual que omitir)."
        ),
    )
    parser.add_argument(
        "--no-empty-cuda-cache",
        action="store_true",
        help="No vaciar caché CUDA entre licitaciones (más rápido; puede peor fragmentación).",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Chunks por forward (más VRAM → más rápido). Default según ModelConfig.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile del backbone del LM (PyTorch 2+; primera corrida más lenta)",
    )
    parser.add_argument(
        "--shard",
        type=int,
        nargs=2,
        metavar=("I", "N"),
        default=None,
        help="Procesar solo licitaciones con hash(tenderId) %% N == I (multi-GPU / multi-job)",
    )
    args = parser.parse_args()

    load_dotenv()
    if not torch.cuda.is_available() and (args.device is None or str(args.device).startswith("cuda")):
        print(
            "Aviso: CUDA no disponible; el embedder en CPU será muy lento.",
            file=sys.stderr,
        )

    bucket = spaces_io.bucket_name()
    if not bucket:
        print("Falta DO_SPACES_BUCKET o SPACES_BUCKET", file=sys.stderr)
        sys.exit(1)
    dataset_prefix = spaces_io.dataset_prefix()
    dataset_key = spaces_io.object_key(dataset_prefix, DEFAULT_S3_DATASET_NAME)
    client = spaces_io.s3_client()
    dataset_path = args.dataset.resolve()
    merged = _load_dataset_local_or_s3(dataset_path, client, bucket, dataset_key)
    if not merged:
        print("Dataset vacío o no se pudo cargar.", file=sys.stderr)
        sys.exit(1)

    local_out = (args.local_out or dataset_path).resolve()
    shard_i, shard_n = (0, 1)
    if args.shard is not None:
        shard_i, shard_n = int(args.shard[0]), int(args.shard[1])
        if shard_n < 1 or not (0 <= shard_i < shard_n):
            print("--shard I N requiere N >= 1 y 0 <= I < N", file=sys.stderr)
            sys.exit(1)

    base_cfg = ModelConfig()
    max_doc_tokens: int | None
    if args.max_doc_tokens is None or args.max_doc_tokens == 0:
        max_doc_tokens = None
    else:
        max_doc_tokens = max(1, int(args.max_doc_tokens))

    cfg = ModelConfig(
        model_id=args.model_id or base_cfg.model_id,
        device=args.device or base_cfg.device,
        dtype=base_cfg.dtype,
        max_len=args.max_len if args.max_len is not None else base_cfg.max_len,
        stride=args.stride if args.stride is not None else base_cfg.stride,
        max_doc_tokens=max_doc_tokens,
        chunk_batch_size=args.chunk_batch_size
        if args.chunk_batch_size is not None
        else base_cfg.chunk_batch_size,
        d_model=base_cfg.d_model,
        n_heads=base_cfg.n_heads,
        ffn_dim=base_cfg.ffn_dim,
        dropout=base_cfg.dropout,
        num_layers=base_cfg.num_layers,
    )

    emb_prefix = spaces_io.pbc_embeddings_prefix_key()
    print(f"Prefijo embeddings: s3://{bucket}/{emb_prefix}")
    existing_emb = frozenset(
        k
        for k in spaces_io.list_object_keys_under_prefix(client, bucket, emb_prefix)
        if k.endswith(".pt")
    )
    print(f"  → {len(existing_emb)} .pt ya listados\n")

    candidates: list[dict] = []
    for row in merged:
        if not isinstance(row, dict):
            continue
        if row.get("pbc_text_extracted") is not True:
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        if not _row_in_shard(tid, shard_i, shard_n):
            continue
        candidates.append(row)

    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    candidate_ids = {id(r) for r in candidates}
    work_total = len(candidates)
    skipped_existing = 0
    for row in candidates:
        stem = _safe_file_stem(str(row.get("tenderId", "")).strip())
        k = spaces_io.pbc_embedding_object_key(stem)
        if k in existing_emb and not args.force:
            skipped_existing += 1

    work_real = sum(
        1
        for row in candidates
        if spaces_io.pbc_embedding_object_key(_safe_file_stem(str(row.get("tenderId", "")).strip()))
        not in existing_emb
        or args.force
    )
    print(
        f"Candidatos (texto OK + shard): {work_total} · "
        f"a procesar con GPU (~{work_real}) · ya en Spaces (~{skipped_existing} sin --force)\n",
    )
    mdt_s = str(cfg.max_doc_tokens) if cfg.max_doc_tokens is not None else "sin límite"
    print(
        f"Modelo: {cfg.model_id} · device={cfg.device} · chunk_batch_size={cfg.chunk_batch_size} "
        f"· max_doc_tokens={mdt_s}\n"
    )

    if args.dry_run:
        print("Dry-run: sin cargar LM ni escribir ficheros.")
        return

    print("Cargando LM + tokenizer…")
    t_build0 = time.monotonic()
    embedder = build_chunk_embedder(cfg)
    embedder.eval()
    if args.compile and hasattr(torch, "compile"):
        backbone = getattr(embedder, "_backbone", None)
        if backbone is not None:
            compiled = torch.compile(
                backbone, mode="reduce-overhead", fullgraph=False
            )
            embedder._backbone = cast(nn.Module, compiled)
            print("torch.compile aplicado al backbone.")
        else:
            print("Aviso: no hay backbone submódulo; omitiendo --compile.", file=sys.stderr)
    print(f"  Listo en {time.monotonic() - t_build0:.1f} s.\n")

    wall0 = time.monotonic()
    work_seconds = 0.0
    work_done_gpu = 0
    done = 0
    cum_ok = 0
    cum_fail = 0
    last_row: dict | None = None
    interrupted = False

    def checkpoint(reason: str) -> None:
        _write_dataset_partial(
            merged,
            local_out=local_out,
            client=client,
            bucket=bucket,
            dataset_prefix=dataset_prefix,
            s3_name=args.s3_output,
            no_upload=args.no_upload,
            dry_run=False,
            candidate_ids=candidate_ids,
            limit=args.limit,
            do_reconcile=False,
        )
        print(f"  [checkpoint] {reason} → {local_out}", flush=True)

    try:
        cp_every = max(0, int(args.checkpoint_every))
        since_cp = 0
        for row in candidates:
            last_row = row
            tid = str(row["tenderId"]).strip()
            stem = _safe_file_stem(tid)
            emb_key = spaces_io.pbc_embedding_object_key(stem)
            txt_key = str(row.get("pbc_txt_s3_key") or "").strip() or spaces_io.pbc_extracted_text_object_key(
                stem
            )

            row.pop("pbc_embedding_skip_reason", None)
            row.pop("pbc_embedding_error", None)

            skip_gpu = emb_key in existing_emb and not args.force

            if skip_gpu:
                row["pbc_embedding_extracted"] = True
                row["pbc_embedding_s3_key"] = emb_key
                row["pbc_embedding_model_id"] = cfg.model_id
                done += 1
                _print_row_done(
                    done, work_total, tid, skipped=True, work_seconds=work_seconds, work_done=work_done_gpu
                )
                since_cp += 1
                if cp_every and since_cp >= cp_every:
                    checkpoint(f"cada {cp_every} filas")
                    since_cp = 0
                continue

            row_t0 = time.monotonic()

            try:
                text_bytes = spaces_io.get_object_bytes(client, bucket, txt_key)
                text = text_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                row["pbc_embedding_extracted"] = False
                row.pop("pbc_embedding_s3_key", None)
                row["pbc_embedding_skip_reason"] = f"txt_read:{type(e).__name__}"
                row["pbc_embedding_error"] = str(e)[:2000]
                cum_fail += 1
                done += 1
                print(f"  [fallo] {tid} · txt_read · {e}", file=sys.stderr)
                since_cp += 1
                if cp_every and since_cp >= cp_every:
                    checkpoint(f"cada {cp_every} filas")
                    since_cp = 0
                continue

            try:
                embs = (
                    forward_text_resolving_cuda_oom(embedder, text)
                    if str(cfg.device).startswith("cuda")
                    else embedder(text)
                )
                n_chunks = int(embs.shape[0])
                y_opt = _training_y_from_row(row)
                blob = _save_embedding_pt(
                    embs=embs,
                    tender_id=tid,
                    model_id=cfg.model_id,
                    max_len=cfg.max_len,
                    stride=cfg.stride,
                    chunk_batch_size=cfg.chunk_batch_size,
                    max_doc_tokens=cfg.max_doc_tokens,
                    y_optional=y_opt,
                )
                spaces_io.put_object_bytes(client, bucket, emb_key, blob)
                existing_emb = existing_emb | {emb_key}
                row["pbc_embedding_extracted"] = True
                row["pbc_embedding_s3_key"] = emb_key
                row["pbc_embedding_n_chunks"] = n_chunks
                row["pbc_embedding_model_id"] = cfg.model_id
                row["pbc_embedding_embedded_at"] = datetime.now(timezone.utc).isoformat()
                cum_ok += 1
            except Exception as e:
                row["pbc_embedding_extracted"] = False
                row.pop("pbc_embedding_s3_key", None)
                row["pbc_embedding_skip_reason"] = f"embed:{type(e).__name__}"
                row["pbc_embedding_error"] = str(e)[:2000]
                cum_fail += 1
                print(f"  [fallo] {tid} · embed · {e}", file=sys.stderr)

            if (
                torch.cuda.is_available()
                and not args.no_empty_cuda_cache
                and str(cfg.device).startswith("cuda")
            ):
                torch.cuda.empty_cache()

            row_dt = time.monotonic() - row_t0
            work_seconds += row_dt
            work_done_gpu += 1
            done += 1
            _print_row_done(
                done, work_total, tid, skipped=False, work_seconds=work_seconds, work_done=work_done_gpu
            )
            since_cp += 1
            if cp_every and since_cp >= cp_every:
                checkpoint(f"cada {cp_every} filas")
                since_cp = 0

        elapsed = time.monotonic() - wall0
        print()
        print("  ---  Fin del bucle  ---")
        if work_seconds > 0 and work_done_gpu > 0:
            print(
                f"      GPU: ~{work_done_gpu / work_seconds * 60.0:.2f} docs/min "
                f"({work_done_gpu} en {work_seconds / 60.0:.1f} min de cómputo)",
            )
        print(
            f"      Reloj total: {elapsed / 60.0:.1f} min · OK {cum_ok} · fallos {cum_fail}",
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C: volcando JSON…", file=sys.stderr)
        if last_row is not None and last_row.get("pbc_embedding_extracted") is not True:
            last_row.setdefault("pbc_embedding_extracted", False)
            last_row["pbc_embedding_skip_reason"] = "interrupted"
            last_row["pbc_embedding_error"] = "Interrupción antes de terminar esta licitación."

    _write_dataset_partial(
        merged,
        local_out=local_out,
        client=client,
        bucket=bucket,
        dataset_prefix=dataset_prefix,
        s3_name=args.s3_output,
        no_upload=args.no_upload,
        dry_run=args.dry_run,
        candidate_ids=candidate_ids,
        limit=args.limit,
        do_reconcile=True,
    )
    if not args.dry_run:
        print(f"\nJSON local: {local_out}")
        if not args.no_upload:
            print(f"S3: s3://{bucket}/{spaces_io.object_key(dataset_prefix, args.s3_output)}")
        n_emb = sum(1 for r in merged if r.get("pbc_embedding_extracted") is True)
        print(f"pbc_embedding_extracted=True: {n_emb} / {len(merged)}")
    if interrupted:
        print("Revisá pbc_embedding_skip_reason=interrupted si aplica.", file=sys.stderr)


if __name__ == "__main__":
    main()
