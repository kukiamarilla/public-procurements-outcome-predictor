"""Extracción de texto y tablas (pdfplumber + camelot) → Markdown por página."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Union

import camelot
import pdfplumber

TableMatrix = List[List[Any]]
PageDict = dict[str, Any]

# Glifos habituales en PDFs (Wingdings/Symbol); ampliar si hace falta
_BULLET_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\uf0d8", "- "),
    ("\uf0b7", "\t- "),
)


class PDFReader:
    def __init__(self, pdf_path: Union[str, Path], *, quiet: bool = False):
        self.pdf_path = str(pdf_path)
        self._quiet = quiet

    def read_pdf(self) -> list[PageDict]:
        """Recorre el PDF y devuelve una lista de dicts por página (texto + tablas)."""
        result: list[PageDict] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_number in range(total_pages):
                result.append(self.read_page(page_number))
        return result

    def read_page(self, page_number: int) -> PageDict:
        """Una página: número (1-based), texto, tablas lattice y stream."""
        camelot_page_number = page_number + 1

        text_content = self.extract_text(page_number)
        lattice_tables_data = self.extract_tables(page_number)
        stream_tables_data = self.extract_stream_tables(page_number)

        return {
            "page": camelot_page_number,
            "text_content": text_content or "",
            "lattice_tables": lattice_tables_data,
            "stream_tables": stream_tables_data,
        }

    def extract_text(self, page_number: int) -> str | None:
        with pdfplumber.open(self.pdf_path) as pdf:
            return pdf.pages[page_number].extract_text()

    def extract_stream_tables(self, page_number: int) -> list[TableMatrix]:
        camelot_page_number = page_number + 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tables = camelot.read_pdf(
                    self.pdf_path,
                    flavor="stream",
                    pages=str(camelot_page_number),
                )
            return [table.df.values.tolist() for table in tables]
        except Exception as e:
            if not self._quiet:
                print(
                    f"Warning: Could not extract stream tables from page {camelot_page_number}: {e}",
                )
            return []

    def extract_tables(self, page_number: int) -> list[TableMatrix]:
        camelot_page_number = page_number + 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tables = camelot.read_pdf(
                    self.pdf_path,
                    flavor="lattice",
                    pages=str(camelot_page_number),
                )
            return [table.df.values.tolist() for table in tables]
        except Exception as e:
            if not self._quiet:
                print(
                    f"Warning: Could not extract lattice tables from page {camelot_page_number}: {e}",
                )
            return []

    def table_matrix_to_markdown(self, table_matrix: TableMatrix, header: bool = True) -> str:
        """Convierte matriz (lista de listas) a tabla Markdown."""
        if not table_matrix or not any(table_matrix):
            return ""
        if header and len(table_matrix) > 0:
            header_row = table_matrix[0]
            data_rows = table_matrix[1:]
            md = (
                "| "
                + " | ".join(
                    str(cell).replace("\n", "<br>") if cell is not None else ""
                    for cell in header_row
                )
                + " |\n"
            )
            md += "| " + " | ".join("---" for _ in header_row) + " |\n"
            for row in data_rows:
                md += (
                    "| "
                    + " | ".join(
                        str(cell).replace("\n", "<br>") if cell is not None else ""
                        for cell in row
                    )
                    + " |\n"
                )
        else:
            num_cols = max(len(row) for row in table_matrix)
            md = "| " + " | ".join(f"col{i + 1}" for i in range(num_cols)) + " |\n"
            md += "| " + " | ".join("---" for _ in range(num_cols)) + " |\n"
            for row in table_matrix:
                padded_row = list(row) + [""] * (num_cols - len(row))
                md += (
                    "| "
                    + " | ".join(
                        str(cell).replace("\n", "<br>") if cell is not None else ""
                        for cell in padded_row
                    )
                    + " |\n"
                )
        return md

    def read_pdf_as_markdown(self) -> str:
        pdf_data = self.read_pdf()
        result = ""
        for page in pdf_data:
            result += f"\n\n## Page {page['page']}"
            result += "\n\n" + page["text_content"]
            lattice_tables = [
                self.table_matrix_to_markdown(table) for table in page["lattice_tables"]
            ]
            stream_tables = [
                self.table_matrix_to_markdown(table) for table in page["stream_tables"]
            ]
            result += "\n\n Lattice Tables:\n\n"
            for i, table in enumerate(lattice_tables):
                result += f"\n\n Table {i + 1}:\n\n"
                result += table + "\n\n"
            result += "\n\n Stream Tables:\n"
            for i, table in enumerate(stream_tables):
                result += f"\n\n Table {i + 1}:\n\n"
                result += table + "\n\n"
            result += "$" * 40 + "\n\n"
        for old, new in _BULLET_REPLACEMENTS:
            result = result.replace(old, new)
        return result
