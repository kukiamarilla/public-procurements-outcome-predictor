"""
ETL: descarga IDs desde la API DNCP (Contrataciones PY) por tender.status y
sube listas + checkpoint a DigitalOcean Spaces (S3).

Equivalente al flujo del script TS de referencia, pero en Python (sin proyecto Node).

Por estado se generan archivos separados **junto a** `procurements.json` (dataset
outcome predictor), p. ej.:
  outcome-predictor/procurements/ids_unsuccessful.json
  outcome-predictor/procurements/ids_cancelled.json
  outcome-predictor/procurements/checkpoint_<estado>.json

Cada ítem incluye "status" (misma convención que el script materialize para complete).

Prefijo bajo el bucket:
  - Por defecto: SPACES_PREFIX (si existe) + outcome-predictor/procurements
    (misma lógica que scripts/once/materialize_procurements_with_status.py)
  - Override opcional: DO_SPACES_DATASET_PREFIX o DO_SPACES_PREFIX_INPUT (ruta completa)

Variables de entorno (cualquiera de los alias):
  DO_SPACES_BUCKET o SPACES_BUCKET
  DO_SPACES_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID o SPACES_ACCESS_KEY
  DO_SPACES_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY o SPACES_SECRET_KEY
  DO_SPACES_REGION o SPACES_REGION
  DO_SPACES_ENDPOINT o SPACES_ENDPOINT

Uso:
  uv run python scripts/etl/fetch_ids_by_status.py unsuccessful
  uv run python scripts/etl/fetch_ids_by_status.py cancelled
  uv run python scripts/etl/fetch_ids_by_status.py both --min 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

DELAY_MS = 1000
RETRY_DELAY_MS = 10000
MAX_RETRIES = 3
BASE_URL = "https://www.contrataciones.gov.py/datos/api/v3/doc/search/processes"
CLASSIFICATION_ID = "72131701"
DEFAULT_MIN_IDS = 1000
# Misma carpeta lógica que DEFAULT_DEST del materialize (sin el nombre del archivo).
DEFAULT_DATASET_PREFIX_REL = "outcome-predictor/procurements"

USER_AGENT = "procurements-outcome-etl/1.0 (research; +https://github.com)"


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def _bucket() -> str:
    return _env("DO_SPACES_BUCKET", "SPACES_BUCKET")


def _dataset_prefix() -> str:
    """Ruta bajo el bucket donde viven procurements.json e ids_*.json."""
    explicit = _env("DO_SPACES_DATASET_PREFIX", "DO_SPACES_PREFIX_INPUT")
    if explicit:
        return explicit.strip().strip("/")
    global_prefix = os.environ.get("SPACES_PREFIX")
    return _object_key(global_prefix, DEFAULT_DATASET_PREFIX_REL)


def _s3_client():
    region = _env("DO_SPACES_REGION", "SPACES_REGION") or "us-east-1"
    endpoint = _env("DO_SPACES_ENDPOINT", "SPACES_ENDPOINT")
    key = _env(
        "DO_SPACES_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "SPACES_ACCESS_KEY",
    )
    secret = _env(
        "DO_SPACES_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "SPACES_SECRET_KEY",
    )
    missing: list[str] = []
    if not endpoint:
        missing.append("DO_SPACES_ENDPOINT o SPACES_ENDPOINT")
    if not key:
        missing.append(
            "DO_SPACES_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID / SPACES_ACCESS_KEY",
        )
    if not secret:
        missing.append(
            "DO_SPACES_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY / SPACES_SECRET_KEY",
        )
    if missing:
        print("Faltan variables en .env:", "; ".join(missing), file=sys.stderr)
        sys.exit(1)
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _object_key(prefix: str | None, filename: str) -> str:
    p = (prefix or "").strip().strip("/")
    f = filename.strip().strip("/")
    return f"{p}/{f}" if p else f


def _normalize_status(rows: list[dict[str, Any]], status_label: str) -> None:
    for row in rows:
        row["status"] = status_label


def _build_url(page: int, tender_status: str) -> str:
    q: dict[str, str] = {
        "items_per_page": "100",
        "tender.items.classification.id": CLASSIFICATION_ID,
        "tender.status": tender_status,
        "page": str(page),
    }
    return f"{BASE_URL}?{urllib.parse.urlencode(q)}"


def _extract_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    cr = record.get("compiledRelease")
    if not isinstance(cr, dict):
        cr = {}
    tender = cr.get("tender")
    tender_id = None
    if isinstance(tender, dict):
        tender_id = tender.get("id")
    if not tender_id:
        tender_id = record.get("ocid")
    if not tender_id:
        return None

    award_ids: list[str] = []
    awards = cr.get("awards")
    if isinstance(awards, list):
        for a in awards:
            if isinstance(a, dict) and a.get("id"):
                award_ids.append(str(a["id"]))

    ocid = cr.get("ocid") if cr else None
    if not ocid:
        ocid = record.get("ocid")

    return {
        "ocid": ocid,
        "tenderId": str(tender_id),
        "awardIds": award_ids,
    }


def _load_checkpoint(
    client: Any,
    bucket: str,
    prefix: str,
    status_label: str,
) -> dict[str, int] | None:
    key = _object_key(prefix, f"checkpoint_{status_label}.json")
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            lp = data.get("lastPage")
            tp = data.get("totalPages")
            if isinstance(lp, int) and isinstance(tp, int):
                return {"lastPage": lp, "totalPages": tp}
    except ClientError:
        pass
    return None


def _save_checkpoint(
    client: Any,
    bucket: str,
    prefix: str,
    status_label: str,
    last_page: int,
    total_pages: int,
) -> None:
    key = _object_key(prefix, f"checkpoint_{status_label}.json")
    body = json.dumps({"lastPage": last_page, "totalPages": total_pages}, indent=2)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )


def _load_existing_ids(
    client: Any,
    bucket: str,
    prefix: str,
    status_label: str,
) -> list[dict[str, Any]]:
    key = _object_key(prefix, f"ids_{status_label}.json")
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except ClientError:
        return []


def _save_ids(
    client: Any,
    bucket: str,
    prefix: str,
    status_label: str,
    ids: list[dict[str, Any]],
) -> None:
    key = _object_key(prefix, f"ids_{status_label}.json")
    body = json.dumps(ids, ensure_ascii=False, indent=2)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def _fetch_page(url: str, page_num: int) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, resp.reason, resp.headers, None
                    )
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code >= 500 and attempt < MAX_RETRIES:
                print(
                    f"  Página {page_num}: HTTP {e.code}, "
                    f"reintento {attempt}/{MAX_RETRIES} en {RETRY_DELAY_MS}ms...",
                )
                time.sleep(RETRY_DELAY_MS / 1000.0)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt >= MAX_RETRIES:
                raise
            print(f"  Página {page_num}: error, reintento {attempt}/{MAX_RETRIES}...")
            time.sleep(RETRY_DELAY_MS / 1000.0)
    raise last_err or RuntimeError("max retries")


def _run_status(
    client: Any,
    bucket: str,
    prefix: str,
    tender_status: str,
    status_label: str,
    min_ids: int,
) -> list[dict[str, Any]]:
    """tender_status: valor API; status_label: sufijo de archivo (sin espacios raros)."""
    checkpoint = _load_checkpoint(client, bucket, prefix, status_label)
    existing = _load_existing_ids(client, bucket, prefix, status_label)
    seen = {str(e["tenderId"]) for e in existing if e.get("tenderId")}
    all_ids: list[dict[str, Any]] = list(existing)

    if len(all_ids) >= min_ids:
        print(
            f"[{status_label}] Ya hay {len(all_ids)} IDs (>={min_ids}); "
            "no se requiere más scraping.",
        )
        return all_ids

    start_page = 1
    total_pages = 1

    if checkpoint:
        if checkpoint["lastPage"] >= checkpoint["totalPages"]:
            if len(all_ids) < min_ids:
                print(
                    f"[{status_label}] Checkpoint al final pero solo {len(all_ids)} IDs "
                    f"(objetivo {min_ids}). No hay más páginas.",
                )
            else:
                print(f"[{status_label}] Checkpoint: corrida previa completada.")
            return all_ids
        start_page = checkpoint["lastPage"] + 1
        total_pages = checkpoint["totalPages"]
        print(
            f"[{status_label}] Reanudando página {start_page}/{total_pages} "
            f"({len(all_ids)} IDs en bucket)",
        )

    # total_pages puede crecer tras la 1ª respuesta (como en el bucle for de JS);
    # por eso no usamos range() fijo.
    page = start_page
    while page <= total_pages:
        url = _build_url(page, tender_status=tender_status)
        data = _fetch_page(url, page)
        records = data.get("records") or []
        if not isinstance(records, list):
            records = []
        pagination = data.get("pagination")

        if pagination and isinstance(pagination, dict) and page == start_page and not checkpoint:
            tp = pagination.get("total_pages")
            if isinstance(tp, int) and tp >= 1:
                total_pages = tp

        new_count = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            extracted = _extract_from_record(record)
            if extracted and extracted["tenderId"] not in seen:
                seen.add(extracted["tenderId"])
                all_ids.append(extracted)
                new_count += 1

        print(
            f"  [{status_label}] Página {page}/{total_pages} "
            f"- +{new_count} nuevos, total: {len(all_ids)}",
        )

        _normalize_status(all_ids, status_label)
        _save_ids(client, bucket, prefix, status_label, all_ids)
        _save_checkpoint(client, bucket, prefix, status_label, page, total_pages)

        if len(all_ids) >= min_ids:
            print(f"[{status_label}] Objetivo {min_ids} alcanzado.")
            break

        if page < total_pages:
            time.sleep(DELAY_MS / 1000.0)
        page += 1

    if len(all_ids) < min_ids:
        print(
            f"[{status_label}] ADVERTENCIA: solo {len(all_ids)} IDs "
            f"(objetivo {min_ids}). Puede faltar volumen en la API para este estado.",
            file=sys.stderr,
        )

    return all_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL DNCP → Spaces: ids por tender.status (unsuccessful / cancelled)",
    )
    parser.add_argument(
        "status",
        choices=("unsuccessful", "cancelled", "both"),
        help="Estado API tender.status o 'both' para ambos",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_IDS,
        dest="min_ids",
        help=f"Mínimo de IDs únicos por estado (default: {DEFAULT_MIN_IDS})",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefijo carpeta en bucket (sobrescribe DO_SPACES_DATASET_PREFIX / SPACES_PREFIX+outcome-predictor/procurements)",
    )
    args = parser.parse_args()

    load_dotenv()
    bucket = _bucket()
    if not bucket:
        print("Falta DO_SPACES_BUCKET o SPACES_BUCKET", file=sys.stderr)
        sys.exit(1)

    prefix = (args.prefix or _dataset_prefix()).strip().strip("/")
    client = _s3_client()

    print("API:", BASE_URL)
    print("Clasificación tender.items.classification.id:", CLASSIFICATION_ID)
    print(f"Prefijo Spaces: {prefix}/")
    print(f"Delay entre requests: {DELAY_MS}ms\n")

    if args.status == "both":
        _run_status(client, bucket, prefix, "unsuccessful", "unsuccessful", args.min_ids)
        _run_status(client, bucket, prefix, "cancelled", "cancelled", args.min_ids)
        print(f"\nListo. s3://{bucket}/{prefix}/ids_unsuccessful.json")
        print(f"        s3://{bucket}/{prefix}/ids_cancelled.json")
    else:
        _run_status(client, bucket, prefix, args.status, args.status, args.min_ids)
        print(f"\nTotal en s3://{bucket}/{prefix}/ids_{args.status}.json")


if __name__ == "__main__":
    main()
