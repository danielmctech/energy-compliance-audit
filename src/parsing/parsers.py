"""
pdf -> markdown parsers for the 02 evaluation (imported by the notebook so
each parser is exercised through the same code path in both places).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pymupdf4llm

try:
    from docling.document_converter import DocumentConverter
except Exception:  # docling is optional
    DocumentConverter = None

try:
    from unstructured.partition.pdf import partition_pdf
except Exception:  # unstructured is optional
    partition_pdf = None

PathLike = Union[str, Path]


def parse_pymupdf4llm(pdf_path: PathLike) -> str:
    return pymupdf4llm.to_markdown(str(pdf_path))


def parse_docling(pdf_path: PathLike) -> str:
    if DocumentConverter is None:
        raise RuntimeError("docling is not installed")
    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_markdown()


def parse_unstructured(pdf_path: PathLike) -> str:
    if partition_pdf is None:
        raise RuntimeError("unstructured is not installed")
    elements = partition_pdf(filename=str(pdf_path))
    return "\n\n".join(str(el) for el in elements)


def available_parsers() -> dict:
    out = {"pymupdf4llm": True, "docling": DocumentConverter is not None,
           "unstructured": partition_pdf is not None}
    return out


PARSERS = {
    "pymupdf4llm": parse_pymupdf4llm,
    "docling": parse_docling,
    "unstructured": parse_unstructured,
}
