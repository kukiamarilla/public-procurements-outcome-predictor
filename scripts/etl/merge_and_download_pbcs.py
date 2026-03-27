"""
1) Lee desde Spaces los tres listados del dataset outcome y los fusiona por tenderId.
2) Por cada licitación intenta bajar el pliego en PDF (solo PDF; si no aplica, skip).
3) Sube cada PBC a Spaces: outcome-predictor/pbcs/pdf/{tender_id_sanitizado}.pdf
4) Guarda JSON enriquecido en local y opcionalmente lo sube al prefijo procurements.
   Antes de escribir, reconcilia pbc_downloaded con un listado completo del prefijo de PDFs en Spaces.
   Si existe un procurements_dataset.json previo (local o S3), preserva pbc_text_* por tenderId (ETL de extracción).

Orden fijo de lectura / merge / cola de descarga:
  unsuccessful → cancelled → complete (procurements.json),
  así al cortar y re-ejecutar los PBC de licitaciones no exitosas se priorizan.

  uv run python scripts/etl/merge_and_download_pbcs.py
  uv run python scripts/etl/merge_and_download_pbcs.py --limit 20 --dry-run
  uv run python scripts/etl/merge_and_download_pbcs.py --omit-doc-docx
  Ctrl+C: escribe igual procurements_dataset.json (local y S3 si aplica) con el progreso hasta ese momento.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

import pbc_simple
import spaces_io

REPO_ROOT = Path(__file__).resolve().parents[2]

# Orden de capas: primera gana en claves duplicadas; define orden de descarga de PDFs.
MERGE_SOURCES_IN_ORDER = (
    "ids_unsuccessful.json",
    "ids_cancelled.json",
    "procurements.json",
)

DEFAULT_S3_OUTPUT = "procurements_dataset.json"
DEFAULT_LOCAL_OUTPUT = REPO_ROOT / "data" / "processed" / "procurements_dataset.json"

PBC_ERROR_MAX_LEN = 8000


def _exception_detail(exc: BaseException, *, extra: str = "") -> str:
    """Texto guardado en JSON (truncado) para reproducir el fallo."""
    head = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    parts = [head]
    if extra.strip():
        parts.append(extra.strip())
    s = "\n".join(parts)
    if len(s) <= PBC_ERROR_MAX_LEN:
        return s
    return f"{s[: PBC_ERROR_MAX_LEN - 24]}\n…[truncado]"


def _is_word_attachment(doc: dict) -> bool:
    """Adjunto declarado como .doc/.docx por título, URL o format (API DNCP)."""
    t = str(doc.get("title") or "").lower().strip()
    if t.endswith((".doc", ".docx")):
        return True
    u = str(doc.get("url") or "").lower().split("?", 1)[0]
    if u.endswith((".doc", ".docx")):
        return True
    fmt = str(doc.get("format") or "").lower()
    if fmt in (
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return True
    if "msword" in fmt or "wordprocessingml" in fmt:
        return True
    return False


def _safe_file_stem(tender_id: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(tender_id).strip(), flags=re.ASCII)
    return (s[:max_len] if s else "unknown").strip("_") or "unknown"


_PRESERVE_PBC_TEXT_KEYS = frozenset(
    {
        "pbc_text_extracted",
        "pbc_txt_s3_key",
        "pbc_text_skip_reason",
        "pbc_text_error",
    }
)


def _load_previous_procurements_dataset_by_tender_id(
    client: object,
    bucket: str,
    prefix: str,
    local_path: Path,
    s3_output_name: str,
) -> dict[str, dict]:
    """
    Filas del último procurements_dataset (local si existe; si no, S3), indexadas por tenderId.
    """
    rows: list[dict] = []
    if local_path.is_file():
        try:
            data = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict)]
        except (OSError, UnicodeError, json.JSONDecodeError):
            rows = []
    if not rows:
        try:
            key = spaces_io.object_key(prefix, s3_output_name)
            raw = spaces_io.get_object_bytes(client, bucket, key)
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict)]
        except Exception:
            pass
    by_tid: dict[str, dict] = {}
    for r in rows:
        tid = str(r.get("tenderId", "")).strip()
        if tid:
            by_tid[tid] = r
    return by_tid


def _apply_preserved_pbc_text_fields(merged: list[dict], prev_by_tid: dict[str, dict]) -> int:
    """Copia pbc_text_* del dataset previo cuando coincide tenderId."""
    if not prev_by_tid:
        return 0
    n = 0
    for row in merged:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid or tid not in prev_by_tid:
            continue
        prev = prev_by_tid[tid]
        touched = False
        for k in _PRESERVE_PBC_TEXT_KEYS:
            if k in prev:
                row[k] = prev[k]
                touched = True
        if touched:
            n += 1
    return n


def _progress_line(done: int, total: int, elapsed_s: float) -> str:
    """Ritmo acumulado desde el inicio del bucle + ETA lineal simple."""
    if elapsed_s <= 0:
        return f"{done}/{total}"
    per_min = done / elapsed_s * 60.0
    remaining = total - done
    eta_min = (remaining / per_min) if per_min > 0 else 0.0
    return f"{done}/{total} | ~{per_min:.1f} lic/min | ETA ~{eta_min:.0f} min"


def _print_batch_summary(
    done: int,
    total: int,
    elapsed_s: float,
    batch_ok: int,
    batch_fail: int,
    cum_ok: int,
    cum_fail: int,
    *,
    is_first: bool,
) -> None:
    """Bloque legible: separador + progreso + stats del batch + acumulado."""
    n_in_batch = batch_ok + batch_fail
    row_from = max(1, done - n_in_batch + 1)
    if not is_first:
        print()
    print(f"  ---  Filas {row_from}–{done} de {total}  ---")
    print(f"      Progreso: {_progress_line(done, total, elapsed_s)}")
    print(f"      Este batch:  OK {batch_ok}  ·  fallos {batch_fail}")
    print(f"      Acumulado:   OK {cum_ok}  ·  fallos {cum_fail}")


def _print_pbc_failure_stderr(row: dict, tid: str, *, max_chars: int = 900) -> None:
    """Imprime el fallo en vivo; el detalle completo sigue en `pbc_error` del JSON."""
    reason = str(row.get("pbc_skip_reason") or "")
    if reason == "dry_run":
        return
    err_raw = str(row.get("pbc_error") or reason).strip()
    lines = err_raw.splitlines()
    head = " ".join(lines[0].split()) if lines else reason
    if len(lines) > 1:
        head += f" (+{len(lines) - 1} líneas; ver JSON: pbc_error)"
    if len(head) > max_chars:
        head = head[: max_chars - 1] + "…"
    tid_disp = tid.strip() or "?"
    print(f"  [fallo] tenderId={tid_disp} · {reason} · {head}", file=sys.stderr)


def _apply_limit_defaults(merged: list[dict], todo_ids: set[int], limit: int) -> None:
    if limit and limit > 0:
        for row in merged:
            if id(row) in todo_ids:
                continue
            row.setdefault("pbc_downloaded", False)
            row.setdefault("pbc_skip_reason", "excluded_by_limit")
            row.setdefault(
                "pbc_error",
                "Excluida por --limit (no procesada en esta corrida).",
            )


def _reconcile_pbc_rows_with_spaces_prefix(
    merged: list[dict],
    client: object,
    bucket: str,
) -> None:
    """
    Fuente de verdad para PBC ya procesados: listado completo del prefijo de PDFs en Spaces.
    Actualiza pbc_downloaded / pbc_s3_key de todas las filas con tenderId, no solo las
    tocadas en la última corrida.
    """
    pdf_prefix = spaces_io.pbc_pdf_prefix_key()
    keys = frozenset(
        spaces_io.list_object_keys_under_prefix(client, bucket, pdf_prefix)
    )
    print(
        f"Reconciliación con prefijo PBC (fuente de verdad): "
        f"{len(keys)} objeto(s) bajo s3://{bucket}/{pdf_prefix}",
    )
    for row in merged:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        stem_key = spaces_io.pbc_pdf_object_key(_safe_file_stem(tid))
        stored = str(row.get("pbc_s3_key") or "").strip()
        # Clave canónica por tenderId o la que ya venía en la fila (subidas viejas / otro stem).
        if stored and stored in keys:
            pdf_key = stored
        elif stem_key in keys:
            pdf_key = stem_key
        else:
            pdf_key = None

        if pdf_key is not None:
            row["pbc_downloaded"] = True
            row["pbc_s3_key"] = pdf_key
            row.pop("pbc_skip_reason", None)
            row.pop("pbc_error", None)
        else:
            was_ok = row.get("pbc_downloaded") is True
            row["pbc_downloaded"] = False
            row.pop("pbc_s3_key", None)
            if was_ok:
                row["pbc_skip_reason"] = "no_pdf_in_spaces"
                row["pbc_error"] = (
                    "El JSON indicaba PBC descargado pero el PDF no está en el prefijo de Spaces."
                )


def _write_dataset_outputs(
    merged: list[dict],
    args: argparse.Namespace,
    local_out: Path,
    client: object,
    bucket: str,
    prefix: str,
    todo_ids: set[int],
    *,
    footer_extra: str = "",
) -> None:
    """Escribe JSON local y, si corresponde, sube el dataset a Spaces."""
    _apply_limit_defaults(merged, todo_ids, args.limit)

    if args.dry_run:
        print("\nDry-run: sin archivos escritos.")
        if footer_extra:
            print(footer_extra, file=sys.stderr)
        return

    _reconcile_pbc_rows_with_spaces_prefix(merged, client, bucket)

    local_out.parent.mkdir(parents=True, exist_ok=True)
    with local_out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nJSON local: {local_out}")

    if not args.no_upload:
        out_key = spaces_io.object_key(prefix, args.s3_output)
        spaces_io.put_json(client, bucket, out_key, merged)
        print(f"S3: s3://{bucket}/{out_key}")

    if footer_extra:
        print(footer_extra, file=sys.stderr)

    ok = sum(1 for r in merged if r.get("pbc_downloaded") is True)
    print(f"pbc_downloaded=True: {ok} / {len(merged)}")


def merge_by_tender_id(layers: list[list[dict]]) -> list[dict]:
    """Primera capa gana en claves; las siguientes solo rellenan huecos None."""
    by_tid: dict[str, dict] = {}
    for rows in layers:
        for row in rows:
            if not isinstance(row, dict):
                continue
            tid = str(row.get("tenderId", "")).strip()
            if not tid:
                continue
            if tid not in by_tid:
                by_tid[tid] = dict(row)
            else:
                base = by_tid[tid]
                for k, v in row.items():
                    if k not in base or base[k] is None:
                        base[k] = v
    return list(by_tid.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge IDs + descarga PBC PDF")
    parser.add_argument("--local-out", type=Path, default=DEFAULT_LOCAL_OUTPUT)
    parser.add_argument(
        "--s3-output",
        default=DEFAULT_S3_OUTPUT,
        help="Nombre del objeto bajo el prefijo del dataset (solo nombre/archivo relativo)",
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--delay", type=float, default=0.75, help="Segundos entre tenders (API)")
    parser.add_argument("--limit", type=int, default=0, help="0 = sin límite")
    parser.add_argument("--dry-run", action="store_true", help="No descarga ni escribe")
    parser.add_argument(
        "--omit-doc-docx",
        action="store_true",
        help="No descargar pliegos cuyo anexo sea .doc o .docx (por título, URL o format de la API)",
    )
    args = parser.parse_args()

    load_dotenv()
    bucket = spaces_io.bucket_name()
    if not bucket:
        print("Falta DO_SPACES_BUCKET o SPACES_BUCKET", file=sys.stderr)
        sys.exit(1)

    prefix = spaces_io.dataset_prefix()
    client = spaces_io.s3_client()

    print("Orden de lectura:", " → ".join(MERGE_SOURCES_IN_ORDER), "\n")

    layers: list[list[dict]] = []
    for name in MERGE_SOURCES_IN_ORDER:
        key = spaces_io.object_key(prefix, name)
        rows = spaces_io.get_json_list(client, bucket, key)
        print(f"{key}: {len(rows)} filas")
        layers.append(rows)

    merged = merge_by_tender_id(layers)
    print(f"Merge único tenderId: {len(merged)} licitaciones")
    print(
        "(Los PDF en Spaces suelen ser menos: sin pliego, errores API, o prefijo distinto en corridas viejas.)\n"
    )

    local_out = args.local_out.resolve()
    prev_by_tid = _load_previous_procurements_dataset_by_tender_id(
        client, bucket, prefix, local_out, args.s3_output
    )
    if prev_by_tid:
        applied = _apply_preserved_pbc_text_fields(merged, prev_by_tid)
        print(
            f"Preservados campos pbc_text_* desde dataset previo: "
            f"{applied} licitación(es) coinciden (de {len(prev_by_tid)} en el archivo anterior).\n"
        )

    todo = merged
    if args.limit and args.limit > 0:
        todo = merged[: args.limit]
    todo_ids = {id(r) for r in todo}

    pdf_prefix = spaces_io.pbc_pdf_prefix_key()
    print(f"Prefijo dataset (JSON): s3://{bucket}/{prefix}/")
    print(f"Prefijo PBC PDF:        s3://{bucket}/{pdf_prefix}")
    print("Listando PDFs existentes …")
    existing_pdf_keys = spaces_io.list_object_keys_under_prefix(client, bucket, pdf_prefix)
    print(
        f"  → {len(existing_pdf_keys)} objetos en prefijo (evita 1 HeadObject por licitación)\n"
    )

    t0 = time.monotonic()
    PROGRESS_EVERY = 5
    batch_ok = 0
    batch_fail = 0
    cum_ok = 0
    cum_fail = 0
    first_batch = True
    interrupted = False
    last_row: dict | None = None

    try:
        for i, row in enumerate(todo):
            last_row = row
            row.pop("pbc_skip_reason", None)
            row.pop("pbc_error", None)
            tid = str(row.get("tenderId", "")).strip()
            need_delay = False

            if not tid:
                row["pbc_downloaded"] = False
                row["pbc_skip_reason"] = "missing_tenderId"
                row["pbc_error"] = "Fila sin tenderId."
            else:
                stem = _safe_file_stem(tid)
                pdf_key = spaces_io.pbc_pdf_object_key(stem)
                row.pop("pbc_local_path", None)

                if pdf_key in existing_pdf_keys:
                    row["pbc_downloaded"] = True
                    row["pbc_s3_key"] = pdf_key
                    row.pop("pbc_skip_reason", None)
                    row.pop("pbc_error", None)
                elif args.dry_run:
                    row["pbc_downloaded"] = False
                    row["pbc_skip_reason"] = "dry_run"
                    row["pbc_error"] = "Dry-run: no se llamó a la API ni a Spaces."
                else:
                    need_delay = True
                    row["pbc_downloaded"] = False
                    try:
                        docs = pbc_simple.fetch_tender_documents(tid)
                        picked = pbc_simple.pick_pdf_pliego(docs)
                        if not picked:
                            row["pbc_downloaded"] = False
                            row["pbc_skip_reason"] = "no_pdf_pliego"
                            n = len(docs) if isinstance(docs, list) else 0
                            row["pbc_error"] = (
                                "Ningún anexo PBC/carta en .pdf/.zip/.rar "
                                f"(documentos en API: {n})."
                            )
                        elif args.omit_doc_docx and _is_word_attachment(picked):
                            row["pbc_downloaded"] = False
                            row["pbc_skip_reason"] = "omit_doc_docx"
                            row["pbc_error"] = (
                                "Omitido: anexo PBC en .doc/.docx (--omit-doc-docx)."
                            )
                        else:
                            pdf_bytes = pbc_simple.download_pdf_bytes(picked)
                            spaces_io.put_pdf_bytes(client, bucket, pdf_key, pdf_bytes)
                            existing_pdf_keys.add(pdf_key)
                            row["pbc_downloaded"] = True
                            row["pbc_s3_key"] = pdf_key
                            row.pop("pbc_skip_reason", None)
                            row.pop("pbc_error", None)
                    except requests.HTTPError as e:
                        row["pbc_downloaded"] = False
                        row["pbc_skip_reason"] = (
                            f"http_{e.response.status_code if e.response else 'err'}"
                        )
                        extra = ""
                        if e.response is not None:
                            try:
                                extra = (
                                    f"url: {e.response.url}\n"
                                    f"body:\n{e.response.text[:2000]}"
                                )
                            except Exception as resp_e:
                                extra = f"(sin body legible: {resp_e!s})"
                        row["pbc_error"] = _exception_detail(e, extra=extra)
                    except requests.RequestException as e:
                        row["pbc_downloaded"] = False
                        row["pbc_skip_reason"] = f"request:{type(e).__name__}"
                        row["pbc_error"] = _exception_detail(e)
                    except ImportError as e:
                        row["pbc_downloaded"] = False
                        row["pbc_skip_reason"] = f"import:{type(e).__name__}"
                        row["pbc_error"] = _exception_detail(e)
                    except (OSError, ValueError, ClientError) as e:
                        row["pbc_downloaded"] = False
                        row["pbc_skip_reason"] = str(e)[:200]
                        row["pbc_error"] = _exception_detail(e)

            if row.get("pbc_downloaded") is True:
                batch_ok += 1
                cum_ok += 1
            else:
                batch_fail += 1
                cum_fail += 1
                _print_pbc_failure_stderr(row, tid)

            done = i + 1
            elapsed = time.monotonic() - t0
            if done % PROGRESS_EVERY == 0 or done == len(todo):
                _print_batch_summary(
                    done,
                    len(todo),
                    elapsed,
                    batch_ok,
                    batch_fail,
                    cum_ok,
                    cum_fail,
                    is_first=first_batch,
                )
                first_batch = False
                batch_ok = 0
                batch_fail = 0

            if need_delay and args.delay > 0 and done < len(todo):
                time.sleep(args.delay)

        elapsed_total = time.monotonic() - t0
        if len(todo) > 0 and elapsed_total > 0:
            avg_per_min = len(todo) / elapsed_total * 60.0
            print()
            print("  ---  Fin del bucle  ---")
            print(
                f"      Promedio: ~{avg_per_min:.1f} lic/min "
                f"({len(todo)} filas en {elapsed_total / 60:.1f} min)"
            )
            print(f"      Totales corrida:  OK {cum_ok}  ·  fallos {cum_fail}\n")

    except KeyboardInterrupt:
        interrupted = True
        print(
            "\nInterrupción (Ctrl+C): volcando el dataset con el progreso hasta ahora…",
            file=sys.stderr,
        )
        if last_row is not None and last_row.get("pbc_downloaded") is not True:
            last_row["pbc_downloaded"] = False
            last_row["pbc_skip_reason"] = "interrupted"
            last_row["pbc_error"] = (
                "Interrupción (Ctrl+C) antes de completar el procesamiento de esta licitación."
            )
        elapsed_partial = time.monotonic() - t0
        print(
            f"  Progreso parcial: OK {cum_ok}  ·  fallos {cum_fail} "
            f"· transcurrido ~{elapsed_partial / 60:.1f} min",
            file=sys.stderr,
        )

    footer = ""
    if interrupted:
        footer = (
            "Salida escrita con merge completo y filas ya actualizadas en esta corrida; "
            "revisá pbc_skip_reason=interrupted en la última fila tocada si aplica."
        )
    _write_dataset_outputs(
        merged, args, local_out, client, bucket, prefix, todo_ids, footer_extra=footer
    )


if __name__ == "__main__":
    main()
