#!/usr/bin/env python3
"""Sincroniza bibliografía y figuras del libro con el shortpaper canónico."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BOOK = ROOT / "libro"
CHAPTERS = BOOK / "Capitulos"


def main() -> None:
    CHAPTERS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PAPER / "shortpaper.bib", BOOK / "references.bib")
    shutil.copytree(PAPER / "figures", BOOK / "figures", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
