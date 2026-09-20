#!/usr/bin/env python3
"""Sincroniza insumos canónicos sin sobrescribir la ampliación del libro."""

from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BOOK = ROOT / "libro"
CHAPTERS = BOOK / "Capitulos"


def main() -> None:
    CHAPTERS.mkdir(parents=True, exist_ok=True)
    # La bibliografía del libro extiende a la del artículo. Solo se inicia desde
    # el archivo canónico cuando todavía no existe una copia enriquecida.
    references = BOOK / "references.bib"
    if not references.exists():
        shutil.copy2(PAPER / "shortpaper.bib", references)
    shutil.copytree(PAPER / "figures", BOOK / "figures", dirs_exist_ok=True)


def generar_fragmentos_traducidos() -> None:
    """Divide la traducción existente sin alterar su redacción."""
    source = (CHAPTERS / "p0010contenido.tex").read_text(encoding="utf-8")
    matches = list(re.finditer(r"^\\chapter\{([^}]*)\}\s*$", source, re.MULTILINE))
    out_dir = CHAPTERS / "shortpaper_fragments"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "Introducción": "introduccion",
        "Definición de la tarea": "definicion_tarea",
        "Método": "metodo",
        "Configuración experimental": "configuracion_experimental",
        "Resultados": "resultados",
        "Discusión": "discusion",
        "Limitaciones": "limitaciones",
        "Conclusión": "conclusion",
    }
    for index, match in enumerate(matches):
        title = match.group(1)
        if title not in names:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[start:end].strip()
        (out_dir / f"{names[title]}.tex").write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    generar_fragmentos_traducidos()
