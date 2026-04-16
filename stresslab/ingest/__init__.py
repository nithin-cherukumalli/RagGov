"""Parsed ingest artifact models and extraction helpers."""

from importlib import import_module

from .models import ParsedDocument, ParsedMetadata, ParsedNode, ParsedTable

_LAZY_EXPORTS = {
    "ExtractedPDF": ".pdf_extract",
    "ExtractedPage": ".pdf_extract",
    "extract_pdf_text": ".pdf_extract",
    "parse_go_order": ".parse_go_order",
}

__all__ = [
    "ExtractedPDF",
    "ExtractedPage",
    "ParsedDocument",
    "ParsedMetadata",
    "ParsedNode",
    "ParsedTable",
    "extract_pdf_text",
    "parse_go_order",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
