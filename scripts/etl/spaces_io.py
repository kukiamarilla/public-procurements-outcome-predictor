"""Utilidades S3/Spaces para scripts ETL (sin dependencia de paquete instalable)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

DEFAULT_DATASET_PREFIX_REL = "outcome-predictor/procurements"
PBC_PDF_PREFIX_REL = "outcome-predictor/pbcs/pdf"


def env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def bucket_name() -> str:
    return env("DO_SPACES_BUCKET", "SPACES_BUCKET")


def dataset_prefix() -> str:
    explicit = env("DO_SPACES_DATASET_PREFIX", "DO_SPACES_PREFIX_INPUT")
    if explicit:
        return explicit.strip().strip("/")
    global_prefix = os.environ.get("SPACES_PREFIX")
    return object_key(global_prefix, DEFAULT_DATASET_PREFIX_REL)


def object_key(prefix: str | None, relative: str) -> str:
    p = (prefix or "").strip().strip("/")
    r = relative.strip().strip("/")
    return f"{p}/{r}" if p else r


def _pbc_pdf_directory_key() -> str:
    """
    Carpeta lógica …/outcome-predictor/pbcs/pdf (sin barra final).

    Debe coincidir con el árbol de `dataset_prefix()`: si los JSON están en
    `{algo}/outcome-predictor/procurements`, los PDF van en
    `{algo}/outcome-predictor/pbcs/pdf`. Antes solo se usaba SPACES_PREFIX, así que con
    DO_SPACES_DATASET_PREFIX=algo/.../procurements y SPACES_PREFIX vacío, los listados
    apuntaban a otro prefijo que el usado al subir los PDFs.
    """
    dp = dataset_prefix().strip().strip("/")
    suffix = "/procurements"
    if dp.endswith(suffix):
        base = dp[: -len(suffix)].rstrip("/")
        if base:
            return f"{base}/pbcs/pdf"
        return "outcome-predictor/pbcs/pdf"
    return object_key(os.environ.get("SPACES_PREFIX"), PBC_PDF_PREFIX_REL).strip("/").rstrip(
        "/"
    )


def pbc_pdf_object_key(safe_tender_stem: str) -> str:
    """Key S3: …/outcome-predictor/pbcs/pdf/{stem}.pdf (alineado con dataset_prefix)."""
    stem = safe_tender_stem.strip().strip("/")
    if not stem:
        stem = "unknown"
    return f"{_pbc_pdf_directory_key()}/{stem}.pdf"


def pbc_pdf_prefix_key() -> str:
    """Prefijo S3 de la carpeta de PDFs, con `/` final (adecuado para `list_objects_v2`)."""
    base = _pbc_pdf_directory_key().rstrip("/")
    return f"{base}/" if base else ""


def _pbc_pbcs_root_directory_key() -> str:
    """Directorio …/pbcs (contiene subcarpetas pdf/ y txt/)."""
    pdf_dir = _pbc_pdf_directory_key().rstrip("/")
    if pdf_dir.endswith("/pdf"):
        return pdf_dir[: -len("/pdf")]
    return pdf_dir


def pbc_extracted_text_object_key(safe_tender_stem: str) -> str:
    """Key S3: …/outcome-predictor/pbcs/txt/{stem}.txt (mismo stem que el PDF en pdf/)."""
    stem = safe_tender_stem.strip().strip("/")
    if not stem:
        stem = "unknown"
    root = _pbc_pbcs_root_directory_key().rstrip("/")
    return f"{root}/txt/{stem}.txt"


def pbc_pbcs_root_prefix_key() -> str:
    """Prefijo `…/pbcs/` para listar todo bajo pbcs (pdf/, txt/, …)."""
    base = _pbc_pbcs_root_directory_key().rstrip("/")
    return f"{base}/" if base else ""


def pbc_extracted_txt_prefix_key() -> str:
    """Prefijo `…/pbcs/txt/` para listar solo extracciones .txt."""
    base = _pbc_pbcs_root_directory_key().rstrip("/")
    return f"{base}/txt/" if base else ""


def pbc_embedding_object_key(safe_tender_stem: str) -> str:
    """Key S3: …/outcome-predictor/pbcs/embeddings/{stem}.pt"""
    stem = safe_tender_stem.strip().strip("/")
    if not stem:
        stem = "unknown"
    root = _pbc_pbcs_root_directory_key().rstrip("/")
    return f"{root}/embeddings/{stem}.pt"


def pbc_embeddings_prefix_key() -> str:
    """Prefijo `…/pbcs/embeddings/` para listar .pt de embeddings por licitación."""
    base = _pbc_pbcs_root_directory_key().rstrip("/")
    return f"{base}/embeddings/" if base else ""


def put_object_bytes(
    client: Any,
    bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_object_bytes(client: Any, bucket: str, key: str) -> bytes:
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def put_text_utf8(client: Any, bucket: str, key: str, text: str) -> None:
    if text is None:
        text = ""
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def list_object_keys_under_prefix(client: Any, bucket: str, prefix: str) -> set[str]:
    """Lista todas las keys bajo `Prefix` (paginado). Una llamada serie de ListObjects, no N HeadObject."""
    p = (prefix or "").strip()
    keys: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=p):
        for obj in page.get("Contents") or []:
            k = obj.get("Key")
            if isinstance(k, str) and k:
                keys.add(k)
    return keys


def object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NotFound", "NoSuchKey"):
            return False
        raise


def put_pdf_bytes(client: Any, bucket: str, key: str, data: bytes) -> None:
    if not data:
        raise ValueError("PDF vacío")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/pdf",
    )


def s3_client():
    load_dotenv()
    region = env("DO_SPACES_REGION", "SPACES_REGION") or "us-east-1"
    endpoint = env("DO_SPACES_ENDPOINT", "SPACES_ENDPOINT")
    key = env(
        "DO_SPACES_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "SPACES_ACCESS_KEY",
    )
    secret = env(
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


def get_json_list(client: Any, bucket: str, key: str) -> list[dict[str, Any]]:
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except ClientError as e:
        print(f"Aviso: no se leyó {key}: {e}", file=sys.stderr)
        return []


def put_json(
    client: Any,
    bucket: str,
    key: str,
    data: list[dict[str, Any]] | dict[str, Any],
) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )
