"""
Descarga de pliego PBC / carta de invitación vía API DNCP.

Soporta anexos .pdf, .zip y .rar (RAR: rarfile + unar o unrar en PATH).
Dentro de ZIP/RAR se admiten .pdf y, si no hay PDF, .doc/.docx convertidos a PDF.
Orden: LibreOffice en modo sin cabeza (si está instalado), luego pandoc + pypandoc
(LaTeX u otro motor PDF suele ser necesario solo para esa vía).
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import requests
from unidecode import unidecode

API_BASE = "https://www.contrataciones.gov.py/datos/api/v3/doc/tender"
USER_AGENT = "procurements-outcome-research/1.0"

_PLIEGO_SUFFIXES = (".pdf", ".zip", ".rar")

# TCSDownloader.select_document: unidecode(documentTypeDetails).lower() in (
#   "pliego de bases y condiciones", "carta de invitacion"
# ). Heurística añadida: pliego electrónico DNCP.
_PBC_DETAILS_EXACT_NORMALIZED = frozenset(
    {
        "pliego de bases y condiciones",
        "carta de invitacion",
        "pliego electronico de bases y condiciones",
    }
)


def _norm_details(details: str | None) -> str:
    """Igual que el original: unidecode + lower (y strip por robustez)."""
    return unidecode(str(details or "")).lower().strip()


def _is_pbc_or_invitation_carta(details_norm: str) -> bool:
    """Equivalente a TCSDownloader.select_document + pliego electrónico."""
    return details_norm in _PBC_DETAILS_EXACT_NORMALIZED


def fetch_tender_documents(tender_id: str, timeout: int = 60) -> list[dict[str, Any]]:
    tid = str(tender_id).strip()
    if not tid:
        return []
    url = f"{API_BASE}/{tid}"
    r = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    tender = data.get("tender")
    if not isinstance(tender, dict):
        return []
    docs = tender.get("documents")
    return docs if isinstance(docs, list) else []


def _title_suffix_ok(title: str) -> bool:
    tl = str(title).lower()
    return any(tl.endswith(s) for s in _PLIEGO_SUFFIXES)


def _is_json_pliego_variant(doc: dict[str, Any]) -> bool:
    """La API duplica el pliego como JSON (/json); no sirve para bajar PDF."""
    fmt = str(doc.get("format") or "").lower()
    if fmt == "application/json":
        return True
    u = str(doc.get("url") or "").lower().rstrip("/")
    if u.endswith("/json"):
        return True
    return "/pliego/" in u and u.split("/")[-1] == "json"


def _url_looks_like_pliego_package(url: str) -> bool:
    u = str(url or "").lower()
    return (
        "documentos/download/pliego" in u
        or "/download/pliego/" in u
        or ("datos/v3/doc" in u and "pliego" in u)
    )


def _can_fetch_pbc_attachment(doc: dict[str, Any]) -> bool:
    """
    El ítem ya calificó por documentTypeDetails como PBC/carta; esto solo filtra
    lo no descargable (JSON del pliego, HTML, sin URL).
    """
    if _is_json_pliego_variant(doc):
        return False
    if not doc.get("url"):
        return False
    fmt = str(doc.get("format") or "").lower()
    if fmt == "text/html":
        return False
    title = str(doc.get("title") or "").strip()
    u = str(doc.get("url") or "").lower()
    if _title_suffix_ok(title):
        return True
    if _url_looks_like_pliego_package(doc.get("url", "")):
        return True
    if fmt in ("application/pdf", "application/zip", "application/x-zip-compressed"):
        return True
    if not fmt or fmt == "application/octet-stream":
        return True
    return bool(title)


def pick_pdf_pliego(documents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Primer documento que cumpla select_document del TCSDownloader original
    (documentTypeDetails normalizado ∈ PBC/carta) y sea descargable en este ETL.
    """
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        try:
            det = doc["documentTypeDetails"]
        except KeyError:
            continue
        details_norm = _norm_details(str(det) if det is not None else "")
        if not _is_pbc_or_invitation_carta(details_norm):
            continue
        if not _can_fetch_pbc_attachment(doc):
            continue
        url = doc.get("url")
        if not url:
            continue
        return doc
    return None


def _is_valid_inner_name(filename: str) -> bool:
    """TCSDownloader.is_valid_document (nombre = basename del miembro)."""
    file_basename = os.path.basename(str(filename).replace("\\", "/"))
    filename_lower = unidecode(file_basename).lower()
    is_pbc_or_invitation = (
        "pliego" in filename_lower
        or "pbc" in filename_lower
        or "carta" in filename_lower
        or "invitacion" in filename_lower
    )
    fl = file_basename.lower()
    is_valid_format = fl.endswith((".pdf", ".doc", ".docx"))
    return is_pbc_or_invitation and is_valid_format


def _download_raw(url: str, timeout: int) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    b = r.content
    if not b:
        raise ValueError("Descarga vacía")
    return b


def _soffice_executable() -> str | None:
    """Ruta a `soffice` si LibreOffice está disponible."""
    w = shutil.which("soffice")
    if w:
        return w
    if sys.platform == "darwin":
        mac = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        if mac.is_file():
            return str(mac)
    return None


def _libreoffice_to_pdf_bytes(path: Path, soffice: str) -> bytes:
    src = path.resolve()
    if not src.is_file():
        raise ValueError("Archivo de entrada inexistente")
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        proc = subprocess.run(
            [
                soffice,
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                "--convert-to",
                "pdf",
                "--outdir",
                str(outdir),
                str(src),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            tail = f": {detail}" if detail else ""
            raise RuntimeError(f"LibreOffice devolvió código {proc.returncode}{tail}")
        expected = outdir / (src.stem + ".pdf")
        if expected.is_file():
            out_pdf = expected
        else:
            pdfs = sorted(outdir.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("LibreOffice no generó ningún PDF en el directorio de salida")
            out_pdf = pdfs[0]
        data = out_pdf.read_bytes()
        if not data:
            raise ValueError("PDF vacío generado por LibreOffice")
        return data


def _pandoc_to_pdf_bytes(path: Path) -> bytes:
    try:
        import pypandoc  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Conversión .doc/.docx con pandoc requiere pypandoc. Ej.: uv sync --extra pbc"
        ) from e
    fd, out_name = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    out_path = Path(out_name)
    try:
        pypandoc.convert_file(str(path.resolve()), "pdf", outputfile=str(out_path))
        data = out_path.read_bytes()
        if not data:
            raise ValueError("La conversión con pandoc produjo un PDF vacío")
        return data
    except OSError as e:
        if "pandoc" in str(e).lower() or "No pandoc" in str(e):
            raise RuntimeError(
                "Pandoc no está en el PATH. macOS: brew install pandoc"
            ) from e
        raise
    finally:
        out_path.unlink(missing_ok=True)


def _convert_office_to_pdf_bytes(path: Path) -> bytes:
    """Convierte .doc/.docx: primero LibreOffice; si falla o no hay, pandoc."""
    errors: list[str] = []
    soffice = _soffice_executable()
    if soffice:
        try:
            return _libreoffice_to_pdf_bytes(path, soffice)
        except Exception as e:
            errors.append(str(e))

    try:
        return _pandoc_to_pdf_bytes(path)
    except ImportError:
        raise
    except Exception as e:
        errors.append(str(e))
        hint = (
            "En macOS suele bastar: brew install --cask libreoffice (y reintentar). "
            "Alternativa: brew install pandoc y un motor LaTeX, p. ej. brew install --cask basictex."
        )
        msg = "No se pudo convertir .doc/.docx a PDF."
        if errors:
            msg += " Detalle: " + " | ".join(errors) + "."
        raise RuntimeError(f"{msg} {hint}") from e


def _file_to_pdf_bytes(path: Path) -> bytes:
    suf = path.suffix.lower()
    if suf == ".pdf":
        data = path.read_bytes()
        if not data:
            raise ValueError("PDF interno vacío")
        return data
    if suf in (".doc", ".docx"):
        return _convert_office_to_pdf_bytes(path)
    raise ValueError(
        f"Formato interno no admitido (.pdf, .doc, .docx); recibido: {path.suffix or '(sin extensión)'}"
    )


def _member_path_safe(member: str, dest_dir: Path) -> Path | None:
    """Evita path traversal; devuelve ruta destino o None si el miembro no es seguro."""
    member = member.replace("\\", "/").strip("/")
    if not member or ".." in member.split("/"):
        return None
    dest = (dest_dir / member).resolve()
    try:
        dest.relative_to(dest_dir.resolve())
    except ValueError:
        return None
    return dest


def _zip_bytes_to_pdf_bytes(raw: bytes) -> bytes:
    """Miembro válido: prioriza .pdf; si no hay, convierte el primero .doc/.docx."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
    except zipfile.BadZipFile as e:
        raise ValueError(f"ZIP inválido: {e!s}") from e
    with zf, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file_list = zf.infolist()
        if not file_list:
            raise ValueError("El archivo ZIP está vacío")
        basenames: list[str] = []
        for info in file_list:
            if info.is_dir():
                continue
            basenames.append(os.path.basename(info.filename.replace("\\", "/")))
        for ext_pass in ((".pdf",), (".doc", ".docx")):
            for info in file_list:
                if info.is_dir():
                    continue
                raw_name = info.filename
                file_basename = os.path.basename(raw_name.replace("\\", "/"))
                if not _is_valid_inner_name(file_basename):
                    continue
                fl = file_basename.lower()
                if not any(fl.endswith(x) for x in ext_pass):
                    continue
                target = _member_path_safe(raw_name, tmp_path)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as out_f:
                    shutil.copyfileobj(src, out_f)
                return _file_to_pdf_bytes(target)
        preview = ", ".join(basenames[:40])
        if len(basenames) > 40:
            preview += "…"
        raise ValueError(
            "No se encontró documento PBC o carta de invitación en ZIP. "
            f"Archivos disponibles: {preview or '(vacío)'}"
        )


def _rar_bytes_to_pdf_bytes(raw: bytes) -> bytes:
    try:
        import rarfile  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "RAR requiere el paquete rarfile. Ej.: uv sync --extra pbc"
        ) from e
    tool = shutil.which("unar") or shutil.which("unrar")
    if not tool:
        raise ValueError(
            "Instalá 'unar' o 'unrar' en el PATH para extraer RAR "
            "(macOS: brew install unar)."
        )
    rarfile.UNRAR_TOOL = tool
    tmp_rar = tempfile.NamedTemporaryFile(suffix=".rar", delete=False)
    try:
        tmp_rar.write(raw)
        tmp_rar.flush()
        rar_path = tmp_rar.name
    finally:
        tmp_rar.close()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with rarfile.RarFile(rar_path) as rf:
                file_list = rf.infolist()
                if not file_list:
                    raise ValueError("El archivo RAR está vacío")
                basenames: list[str] = []
                for info in file_list:
                    if info.isdir():
                        continue
                    basenames.append(os.path.basename(info.filename.replace("\\", "/")))
                for ext_pass in ((".pdf",), (".doc", ".docx")):
                    for info in file_list:
                        if info.isdir():
                            continue
                        raw_name = info.filename
                        file_basename = os.path.basename(raw_name.replace("\\", "/"))
                        if not _is_valid_inner_name(file_basename):
                            continue
                        fl = file_basename.lower()
                        if not any(fl.endswith(x) for x in ext_pass):
                            continue
                        dest_file = _member_path_safe(raw_name, tmp_path)
                        if dest_file is None:
                            continue
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        rf.extract(info, path=tmp_path)
                        if dest_file.is_file():
                            return _file_to_pdf_bytes(dest_file)
                preview = ", ".join(basenames[:40])
                if len(basenames) > 40:
                    preview += "…"
                raise ValueError(
                    "No se encontró documento PBC o carta de invitación en RAR. "
                    f"Archivos disponibles: {preview or '(vacío)'}"
                )
    finally:
        Path(rar_path).unlink(missing_ok=True)


def _normalize_downloaded_to_pdf_bytes(raw: bytes, title: str) -> bytes:
    """
    La API puede marcar `format: application/pdf` pero enviar un ZIP (pliego electrónico).
    """
    if raw.startswith(b"%PDF"):
        return raw
    if len(raw) >= 2 and raw[:2] == b"PK":
        return _zip_bytes_to_pdf_bytes(raw)
    if raw.startswith(b"Rar!"):
        return _rar_bytes_to_pdf_bytes(raw)
    tl = str(title).lower()
    if tl.endswith(".zip"):
        return _zip_bytes_to_pdf_bytes(raw)
    if tl.endswith(".rar"):
        return _rar_bytes_to_pdf_bytes(raw)
    if tl.endswith(".pdf"):
        return raw
    try:
        return _zip_bytes_to_pdf_bytes(raw)
    except (ValueError, zipfile.BadZipFile, OSError):
        return raw


def download_pdf_bytes(doc: dict[str, Any], timeout: int = 120) -> bytes:
    """
    Descarga el anexo y devuelve bytes de PDF (incluye normalización desde ZIP/RAR).
    """
    url = str(doc["url"])
    title = str(doc.get("title", "doc"))
    raw = _download_raw(url, timeout)
    return _normalize_downloaded_to_pdf_bytes(raw, title)


def download_pdf(doc: dict[str, Any], dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(download_pdf_bytes(doc, timeout=timeout))
