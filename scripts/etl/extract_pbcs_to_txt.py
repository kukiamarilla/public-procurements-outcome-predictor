"""
Extrae texto/tablas de los PBC en PDF (Spaces) y sube un .txt por licitación.

- Texto en Spaces: …/outcome-predictor/pbcs/txt/{stem}.txt
- Actualiza procurements_dataset.json (local + S3) con pbc_text_extracted / pbc_txt_s3_key
  y reconciliación contra el prefijo pbcs/txt/ (misma idea que PDF + pbc_downloaded).
- Progreso fila a fila; ritmo y ETA solo cuentan extracciones reales (no .txt ya en Spaces).
  --limit con excluded_by_limit; Ctrl+C guarda dataset + reconciliación.

Requiere: uv sync --extra pdf

  uv run python scripts/etl/extract_pbcs_to_txt.py
  uv run python scripts/etl/extract_pbcs_to_txt.py --limit 10 --dry-run
  uv run python scripts/etl/extract_pbcs_to_txt.py --force
  uv run python scripts/etl/extract_pbcs_to_txt.py --no-upload
  uv run python scripts/etl/extract_pbcs_to_txt.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import spaces_io

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "procurements_dataset.json"
DEFAULT_S3_DATASET_NAME = "procurements_dataset.json"


def _worker_run_extract(payload: dict[str, str]) -> dict[str, Any]:
    """
    Ejecuta en proceso hijo: bajar PDF, extraer, subir .txt.
    Debe ser top-level para multiprocessing (spawn).
    """
    import tempfile
    from pathlib import Path as PathLocal

    from dotenv import load_dotenv

    load_dotenv()
    import spaces_io as sio
    from doc_extract import PDFReader

    tid = payload["tender_id"]
    bucket = payload["bucket"]
    pdf_key = payload["pdf_key"]
    txt_key = payload["txt_key"]
    t0 = time.monotonic()
    try:
        pdf_bytes = sio.get_object_bytes(sio.s3_client(), bucket, pdf_key)
    except Exception as e:
        return {
            "tender_id": tid,
            "pbc_text_extracted": False,
            "pbc_txt_s3_key": None,
            "pbc_text_skip_reason": f"pdf_read:{type(e).__name__}",
            "pbc_text_error": str(e)[:2000],
            "seconds": time.monotonic() - t0,
        }
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        md = PDFReader(tmp_path).read_pdf_as_markdown()
        sio.put_text_utf8(sio.s3_client(), bucket, txt_key, md)
        return {
            "tender_id": tid,
            "pbc_text_extracted": True,
            "pbc_txt_s3_key": txt_key,
            "pbc_text_skip_reason": None,
            "pbc_text_error": None,
            "seconds": time.monotonic() - t0,
        }
    except Exception as e:
        return {
            "tender_id": tid,
            "pbc_text_extracted": False,
            "pbc_txt_s3_key": None,
            "pbc_text_skip_reason": f"extract:{type(e).__name__}",
            "pbc_text_error": str(e)[:2000],
            "seconds": time.monotonic() - t0,
        }
    finally:
        if tmp_path:
            PathLocal(tmp_path).unlink(missing_ok=True)


def _apply_worker_result(row: dict, res: dict[str, Any]) -> None:
    row.pop("pbc_text_skip_reason", None)
    row.pop("pbc_text_error", None)
    if res.get("pbc_text_extracted") is True:
        row["pbc_text_extracted"] = True
        row["pbc_txt_s3_key"] = res.get("pbc_txt_s3_key")
    else:
        row["pbc_text_extracted"] = False
        row.pop("pbc_txt_s3_key", None)
        if res.get("pbc_text_skip_reason"):
            row["pbc_text_skip_reason"] = res["pbc_text_skip_reason"]
        if res.get("pbc_text_error"):
            row["pbc_text_error"] = res["pbc_text_error"]


def _safe_file_stem(tender_id: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(tender_id).strip(), flags=re.ASCII)
    return (s[:max_len] if s else "unknown").strip("_") or "unknown"


def _print_text_failure_stderr(row: dict, tid: str, *, max_chars: int = 900) -> None:
    reason = str(row.get("pbc_text_skip_reason") or "")
    if reason == "dry_run":
        return
    err_raw = str(row.get("pbc_text_error") or reason).strip()
    lines = err_raw.splitlines()
    head = " ".join(lines[0].split()) if lines else reason
    if len(lines) > 1:
        head += f" (+{len(lines) - 1} líneas; ver JSON: pbc_text_error)"
    if len(head) > max_chars:
        head = head[: max_chars - 1] + "…"
    tid_disp = tid.strip() or "?"
    print(f"  [fallo] tenderId={tid_disp} · {reason} · {head}", file=sys.stderr)


def _apply_extract_limit_defaults(
    merged: list[dict],
    candidate_ids: set[int],
    limit: int,
) -> None:
    if not limit or limit <= 0:
        return
    for row in merged:
        if not isinstance(row, dict):
            continue
        if row.get("pbc_downloaded") is not True:
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        if id(row) in candidate_ids:
            continue
        row.setdefault("pbc_text_extracted", False)
        row.setdefault("pbc_text_skip_reason", "excluded_by_limit")
        row.setdefault(
            "pbc_text_error",
            "Excluida por --limit en extract_pbcs_to_txt (no procesada en esta corrida).",
        )


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


def _reconcile_pbc_text_rows_with_spaces_prefix(
    merged: list[dict],
    client: object,
    bucket: str,
) -> None:
    """Fuente de verdad: objetos .txt bajo pbcs/txt/."""
    txt_prefix = spaces_io.pbc_extracted_txt_prefix_key()
    keys = frozenset(
        spaces_io.list_object_keys_under_prefix(client, bucket, txt_prefix)
    )
    keys = frozenset(k for k in keys if k.endswith(".txt"))
    print(
        f"Reconciliación texto PBC (fuente de verdad): "
        f"{len(keys)} .txt bajo s3://{bucket}/{txt_prefix}",
    )
    for row in merged:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        stem_key = spaces_io.pbc_extracted_text_object_key(_safe_file_stem(tid))
        stored = str(row.get("pbc_txt_s3_key") or "").strip()
        if stored and stored in keys:
            txt_key = stored
        elif stem_key in keys:
            txt_key = stem_key
        else:
            txt_key = None

        if txt_key is not None:
            row["pbc_text_extracted"] = True
            row["pbc_txt_s3_key"] = txt_key
            row.pop("pbc_text_skip_reason", None)
            row.pop("pbc_text_error", None)
        else:
            was_ok = row.get("pbc_text_extracted") is True
            row["pbc_text_extracted"] = False
            row.pop("pbc_txt_s3_key", None)
            if was_ok:
                row["pbc_text_skip_reason"] = "no_txt_in_spaces"
                row["pbc_text_error"] = (
                    "El JSON indicaba texto extraído pero el .txt no está en el prefijo de Spaces."
                )


def _write_dataset(
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
) -> None:
    _apply_extract_limit_defaults(merged, candidate_ids, limit)

    if dry_run:
        print("\nDry-run: sin escribir procurements_dataset.json.")
        return
    _reconcile_pbc_text_rows_with_spaces_prefix(merged, client, bucket)
    local_out.parent.mkdir(parents=True, exist_ok=True)
    with local_out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nJSON local: {local_out}")
    if not no_upload:
        out_key = spaces_io.object_key(dataset_prefix, s3_name)
        spaces_io.put_json(client, bucket, out_key, merged)
        print(f"S3: s3://{bucket}/{out_key}")
    n_txt = sum(1 for r in merged if r.get("pbc_text_extracted") is True)
    print(f"pbc_text_extracted=True: {n_txt} / {len(merged)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF PBC → .txt en Spaces + actualiza procurements_dataset.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="procurements_dataset.json de entrada (local; si no existe, se lee de S3)",
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=None,
        help="Salida JSON local (por defecto igual que --dataset)",
    )
    parser.add_argument(
        "--s3-output",
        default=DEFAULT_S3_DATASET_NAME,
        help="Nombre del objeto dataset bajo el prefijo de procurements",
    )
    parser.add_argument("--no-upload", action="store_true", help="No subir el JSON a Spaces")
    parser.add_argument("--limit", type=int, default=0, help="0 = todas las filas con PDF")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vuelve a extraer aunque ya exista el .txt en Spaces",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Procesos en paralelo para extracción (1 = secuencial; p. ej. 4–8 según CPU y RAM)",
    )
    args = parser.parse_args()

    try:
        from doc_extract import PDFReader
    except ImportError:
        print(
            "Falta el extra pdf (pdfplumber/camelot). Ej.: uv sync --extra pdf",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    load_dotenv()
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

    txt_prefix = spaces_io.pbc_extracted_txt_prefix_key()
    print(f"Prefijo extracciones: s3://{bucket}/{txt_prefix}")
    under_txt = spaces_io.list_object_keys_under_prefix(client, bucket, txt_prefix)
    existing_txt = frozenset(k for k in under_txt if k.endswith(".txt"))
    print(f"  → {len(existing_txt)} .txt en pbcs/txt/\n")

    candidates: list[dict] = []
    for row in merged:
        if not isinstance(row, dict):
            continue
        if row.get("pbc_downloaded") is not True:
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        candidates.append(row)

    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    candidate_ids = {id(r) for r in candidates}

    work_total = 0
    for row in candidates:
        tid_w = str(row["tenderId"]).strip()
        stem_w = _safe_file_stem(tid_w)
        txt_w = spaces_io.pbc_extracted_text_object_key(stem_w)
        if txt_w in existing_txt and not args.force:
            continue
        work_total += 1

    t0 = time.monotonic()
    work_seconds = 0.0
    work_done = 0
    cum_ok = 0
    cum_fail = 0
    last_row: dict | None = None
    interrupted = False
    row_by_tid = {str(r["tenderId"]).strip(): r for r in candidates}
    workers = max(1, args.workers)
    pool_wall_seconds = 0.0

    def _finish_row_print(done_idx: int, tid: str, skip_existing: bool, row: dict) -> None:
        nonlocal cum_ok, cum_fail
        if row.get("pbc_text_extracted") is True:
            cum_ok += 1
        else:
            cum_fail += 1
            _print_text_failure_stderr(row, tid)
        total = len(candidates)
        if skip_existing:
            print(f"  [{done_idx}/{total}] {tid} · ya en Spaces (fuera del ritmo)")
        else:
            ok_txt = "OK" if row.get("pbc_text_extracted") is True else "fallo"
            print(f"  [{done_idx}/{total}] {tid} · {ok_txt}")
            if work_total > 0:
                if workers > 1 and pool_wall_seconds > 0:
                    rate = work_done / pool_wall_seconds * 60.0
                    rem = work_total - work_done
                    eta_min = (
                        (rem / (work_done / pool_wall_seconds)) / 60.0 if work_done > 0 else 0.0
                    )
                    print(
                        f"      Extracción: {work_done}/{work_total} · "
                        f"~{rate:.1f} lic/min (paralelo) · ETA ~{eta_min:.0f} min",
                    )
                elif work_seconds > 0:
                    rate = work_done / work_seconds * 60.0
                    rem = work_total - work_done
                    eta_min = (rem / (work_done / work_seconds)) / 60.0 if work_done > 0 else 0.0
                    print(
                        f"      Extracción: {work_done}/{work_total} · "
                        f"~{rate:.1f} lic/min · ETA ~{eta_min:.0f} min",
                    )

    try:
        if workers > 1 and not args.dry_run:
            print(f"Modo paralelo: {workers} workers (procesos).\n")
            pending_payloads: list[dict[str, str]] = []
            done_seq = 0
            for row in candidates:
                last_row = row
                tid = str(row["tenderId"]).strip()
                stem = _safe_file_stem(tid)
                pdf_key = str(row.get("pbc_s3_key") or "").strip() or spaces_io.pbc_pdf_object_key(
                    stem
                )
                txt_key = spaces_io.pbc_extracted_text_object_key(stem)
                row.pop("pbc_text_skip_reason", None)
                row.pop("pbc_text_error", None)
                skip_existing = txt_key in existing_txt and not args.force
                if skip_existing:
                    row["pbc_text_extracted"] = True
                    row["pbc_txt_s3_key"] = txt_key
                    done_seq += 1
                    _finish_row_print(done_seq, tid, True, row)
                else:
                    pending_payloads.append(
                        {
                            "tender_id": tid,
                            "bucket": bucket,
                            "pdf_key": pdf_key,
                            "txt_key": txt_key,
                        }
                    )

            if pending_payloads:
                pool_t0 = time.monotonic()
                executor = ProcessPoolExecutor(max_workers=workers)
                pool_aborted = False
                try:
                    future_map = {
                        executor.submit(_worker_run_extract, p): p for p in pending_payloads
                    }
                    for fut in as_completed(future_map):
                        last_row = row_by_tid[future_map[fut]["tender_id"]]
                        try:
                            res = fut.result()
                        except Exception as e:
                            tid_e = future_map[fut]["tender_id"]
                            res = {
                                "tender_id": tid_e,
                                "pbc_text_extracted": False,
                                "pbc_text_skip_reason": f"worker:{type(e).__name__}",
                                "pbc_text_error": str(e)[:2000],
                            }
                        tid_r = str(res["tender_id"])
                        row_r = row_by_tid[tid_r]
                        _apply_worker_result(row_r, res)
                        work_done += 1
                        pool_wall_seconds = time.monotonic() - pool_t0
                        done_seq += 1
                        _finish_row_print(done_seq, tid_r, False, row_r)
                except KeyboardInterrupt:
                    pool_aborted = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    if not pool_aborted:
                        executor.shutdown(wait=True)
                work_seconds = time.monotonic() - pool_t0
        else:
            for i, row in enumerate(candidates):
                last_row = row
                tid = str(row["tenderId"]).strip()
                stem = _safe_file_stem(tid)
                pdf_key = str(row.get("pbc_s3_key") or "").strip() or spaces_io.pbc_pdf_object_key(
                    stem
                )
                txt_key = spaces_io.pbc_extracted_text_object_key(stem)

                row.pop("pbc_text_skip_reason", None)
                row.pop("pbc_text_error", None)

                skip_existing = txt_key in existing_txt and not args.force
                row_t0 = time.monotonic()

                if skip_existing:
                    row["pbc_text_extracted"] = True
                    row["pbc_txt_s3_key"] = txt_key
                elif args.dry_run:
                    row["pbc_text_extracted"] = False
                    row.pop("pbc_txt_s3_key", None)
                    row["pbc_text_skip_reason"] = "dry_run"
                    row["pbc_text_error"] = "Dry-run: no se extrajo ni se subió texto."
                else:
                    try:
                        pdf_bytes = spaces_io.get_object_bytes(client, bucket, pdf_key)
                    except Exception as e:
                        row["pbc_text_extracted"] = False
                        row.pop("pbc_txt_s3_key", None)
                        row["pbc_text_skip_reason"] = f"pdf_read:{type(e).__name__}"
                        row["pbc_text_error"] = str(e)[:2000]
                        pdf_bytes = None

                    if pdf_bytes is not None:
                        tmp_path: str | None = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                tmp.write(pdf_bytes)
                                tmp_path = tmp.name
                            md = PDFReader(tmp_path).read_pdf_as_markdown()
                            spaces_io.put_text_utf8(client, bucket, txt_key, md)
                            existing_txt = existing_txt | {txt_key}
                            row["pbc_text_extracted"] = True
                            row["pbc_txt_s3_key"] = txt_key
                        except Exception as e:
                            row["pbc_text_extracted"] = False
                            row.pop("pbc_txt_s3_key", None)
                            row["pbc_text_skip_reason"] = f"extract:{type(e).__name__}"
                            row["pbc_text_error"] = str(e)[:2000]
                        finally:
                            if tmp_path:
                                Path(tmp_path).unlink(missing_ok=True)

                row_dt = time.monotonic() - row_t0
                if not skip_existing:
                    work_seconds += row_dt
                    work_done += 1

                done = i + 1
                _finish_row_print(done, tid, skip_existing, row)

        elapsed_total = time.monotonic() - t0
        if len(candidates) > 0:
            print()
            print("  ---  Fin del bucle  ---")
            print(
                f"      Candidatos: {len(candidates)} · filas con trabajo real: {work_total} "
                f"(reloj total {elapsed_total / 60:.1f} min)",
            )
            if workers > 1 and not args.dry_run and work_seconds > 0 and work_total > 0:
                print(
                    f"      Ritmo agregado (paralelo, reloj único): "
                    f"~{work_total / work_seconds * 60:.1f} lic/min "
                    f"({work_total} extracciones en {work_seconds / 60:.1f} min)",
                )
            elif work_seconds > 0:
                print(
                    f"      Ritmo solo extracción: ~{work_done / work_seconds * 60:.1f} lic/min "
                    f"({work_total} filas en {work_seconds / 60:.1f} min de trabajo)",
                )
            print(f"      Totales:  OK {cum_ok}  ·  fallos {cum_fail}\n")

    except KeyboardInterrupt:
        interrupted = True
        print(
            "\nInterrupción (Ctrl+C): volcando el dataset con el progreso de texto hasta ahora…",
            file=sys.stderr,
        )
        if workers <= 1 and last_row is not None and last_row.get("pbc_text_extracted") is not True:
            last_row["pbc_text_extracted"] = False
            last_row.pop("pbc_txt_s3_key", None)
            last_row["pbc_text_skip_reason"] = "interrupted"
            last_row["pbc_text_error"] = (
                "Interrupción (Ctrl+C) antes de completar la extracción de texto de esta licitación."
            )
        elapsed_partial = time.monotonic() - t0
        extra = ""
        if work_total > 0:
            extra = f" · extracción {work_done}/{work_total}"
        print(
            f"  Progreso parcial: OK {cum_ok}  ·  fallos {cum_fail}{extra} "
            f"· reloj ~{elapsed_partial / 60:.1f} min",
            file=sys.stderr,
        )

    footer = ""
    if interrupted:
        footer = (
            "Salida escrita con dataset completo y reconciliación pbcs/txt/; "
            "revisá pbc_text_skip_reason=interrupted si aplica."
        )

    if footer:
        print(footer, file=sys.stderr)

    _write_dataset(
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
    )


if __name__ == "__main__":
    main()
