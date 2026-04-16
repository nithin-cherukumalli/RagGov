"""Minimal PDF text extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedPDF:
    page_count: int
    pages: list[ExtractedPage]


def extract_pdf_text(path: Path) -> ExtractedPDF:
    """Extract page-level text from a PDF."""

    pages: list[ExtractedPage] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(
                ExtractedPage(
                    page_number=page.page_number,
                    text=page.extract_text() or "",
                )
            )
    return ExtractedPDF(page_count=len(pages), pages=pages)
