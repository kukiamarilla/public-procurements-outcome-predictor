"""
One-off: lee tagging/ids/ids.json desde DO Spaces y escribe
outcome-predictor/procurements/procurements.json con "status": "complete" por ítem.

No modifica el objeto fuente. Requiere .env con SPACES_* (ver .env.example).

  uv run python scripts/once/materialize_procurements_with_status.py
  uv run python scripts/once/materialize_procurements_with_status.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

DEFAULT_SOURCE = "tagging/ids/ids.json"
DEFAULT_DEST = "outcome-predictor/procurements/procurements.json"
DEFAULT_STATUS = "complete"


def _object_key(prefix: str | None, relative: str) -> str:
    p = (prefix or "").strip().strip("/")
    r = relative.strip().strip("/")
    return f"{p}/{r}" if p else r


def _s3_client():
    load_dotenv()
    region = os.environ.get("SPACES_REGION")
    endpoint = os.environ.get("SPACES_ENDPOINT")
    key = os.environ.get("SPACES_ACCESS_KEY")
    secret = os.environ.get("SPACES_SECRET_KEY")
    missing = [n for n, v in [
        ("SPACES_REGION", region),
        ("SPACES_ENDPOINT", endpoint),
        ("SPACES_ACCESS_KEY", key),
        ("SPACES_SECRET_KEY", secret),
    ] if not v]
    if missing:
        print("Faltan variables de entorno en .env:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _read_json_array(client, bucket: str, key: str) -> list[dict[str, Any]]:
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        print(f"No se pudo leer s3://{bucket}/{key}: {e}", file=sys.stderr)
        sys.exit(1)
    body = obj["Body"].read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, list):
        print("Se esperaba un array JSON en la raíz.", file=sys.stderr)
        sys.exit(1)
    return data


def _with_status(items: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            print(f"Elemento {i} no es un objeto JSON; abortando.", file=sys.stderr)
            sys.exit(1)
        new_row = {**row, "status": status}
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Materializa procurements.json con status desde ids.json")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Key del objeto fuente")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Key del objeto destino")
    parser.add_argument("--status", default=DEFAULT_STATUS, help='Valor del campo "status"')
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar cantidad; no escribir")
    args = parser.parse_args()

    load_dotenv()
    bucket = os.environ.get("SPACES_BUCKET")
    if not bucket:
        print("Falta SPACES_BUCKET en .env", file=sys.stderr)
        sys.exit(1)
    prefix = os.environ.get("SPACES_PREFIX")

    source_key = _object_key(prefix, args.source)
    dest_key = _object_key(prefix, args.dest)

    client = _s3_client()
    items = _read_json_array(client, bucket, source_key)
    enriched = _with_status(items, args.status)

    payload = json.dumps(enriched, ensure_ascii=False, indent=2)
    print(f"Fuente: s3://{bucket}/{source_key} ({len(items)} ítems)")
    print(f"Destino: s3://{bucket}/{dest_key}")

    if args.dry_run:
        print("Dry-run: no se escribió nada.")
        return

    client.put_object(
        Bucket=bucket,
        Key=dest_key,
        Body=payload.encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )
    print("Escrito OK.")


if __name__ == "__main__":
    main()
