"""Tests for stresslab ingest models."""

from __future__ import annotations

import importlib
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from stresslab.ingest.models import ParsedDocument, ParsedNode, ParsedTable


def test_ingest_package_does_not_import_pdf_extract_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "stresslab.ingest", raising=False)
    monkeypatch.delitem(sys.modules, "stresslab.ingest.pdf_extract", raising=False)

    ingest = importlib.import_module("stresslab.ingest")

    assert "stresslab.ingest.pdf_extract" not in sys.modules

    extracted_pdf = ingest.ExtractedPDF
    extracted_page = ingest.ExtractedPage
    extract_pdf_text = ingest.extract_pdf_text

    assert "stresslab.ingest.pdf_extract" in sys.modules
    assert extracted_pdf.__name__ == "ExtractedPDF"
    assert extracted_page.__name__ == "ExtractedPage"
    assert extract_pdf_text.__name__ == "extract_pdf_text"


def test_parsed_document_accepts_required_lineage_fields() -> None:
    table = ParsedTable(
        table_id="tbl-1",
        page=3,
        title="Distribution",
        headers=["Office", "Copies"],
        rows=[["HQ", "2"], ["Field", "4"]],
    )
    node = ParsedNode(
        node_id="node-1",
        label="Section 1",
        text="Introductory clause.",
        page_start=1,
        page_end=2,
        parent_node_id="root",
        section_path=["Preamble", "Section 1"],
    )

    document = ParsedDocument(
        doc_id="doc-1",
        source_path="/tmp/doc-1.json",
        title="Order on distribution",
        abstract="A short summary.",
        department="Revenue Department",
        go_number="G.O. 12",
        issued_date="2026-04-15",
        references=["Rule 1", "Rule 2"],
        nodes=[node],
        tables=[table],
        distribution=["Collector", "Treasury"],
    )

    assert document.doc_id == "doc-1"
    assert document.nodes[0].parent_node_id == "root"
    assert document.tables[0].rows[0] == ["HQ", "2"]
    assert document.model_dump()["distribution"] == ["Collector", "Treasury"]


def test_parsed_node_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ParsedNode(
            node_id="node-1",
            label="Section 1",
            text="Body",
            page_start=1,
            page_end=1,
            unexpected="value",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("page_start", "page_end"),
    [
        (0, 1),
        (1, 0),
        (-1, 1),
        (2, 1),
    ],
)
def test_parsed_node_rejects_invalid_page_range(page_start: int, page_end: int) -> None:
    with pytest.raises(ValidationError):
        ParsedNode(
            node_id="node-1",
            label="Section 1",
            text="Body",
            page_start=page_start,
            page_end=page_end,
        )


@pytest.mark.parametrize("page", [0, -1])
def test_parsed_table_rejects_non_positive_page(page: int) -> None:
    with pytest.raises(ValidationError):
        ParsedTable(
            table_id="tbl-1",
            page=page,
            title="Distribution",
        )


def test_parsed_document_rejects_conflicting_metadata() -> None:
    with pytest.raises(ValidationError):
        ParsedDocument(
            doc_id="doc-1",
            source_path="/tmp/doc-1.json",
            title="Order on distribution",
            abstract="A short summary.",
            department="Revenue Department",
            go_number="G.O. 12",
            issued_date="2026-04-15",
            references=["Rule 1", "Rule 2"],
            distribution=["Collector", "Treasury"],
            metadata={
                "source_path": "/tmp/other.json",
                "title": "Order on distribution",
                "abstract": "A short summary.",
                "department": "Revenue Department",
                "go_number": "G.O. 12",
                "issued_date": "2026-04-15",
                "references": ["Rule 1", "Rule 2"],
                "distribution": ["Collector", "Treasury"],
            },
        )


def test_extract_pdf_text_reads_ms1() -> None:
    from stresslab.ingest import extract_pdf_text

    result = extract_pdf_text(Path("tests/Data/2011SE_MS1.PDF"))

    assert result.page_count == 1
    assert "Smt.T.Jyothi" in result.pages[0].text


def test_parse_go_order_preserves_rule_hierarchy_for_ms20() -> None:
    from stresslab.ingest import parse_go_order

    document = parse_go_order(Path("tests/Data/2011SE_MS20.PDF"))

    assert document.doc_id == "2011SE_MS20"
    assert "Right of Children to Free and Compulsory Education Rules" in document.title
    assert document.department == "SCHOOL EDUCATION (P.E .PROG.I) DEPARTMENT"
    assert document.go_number == "GO.Ms.No. 20"
    assert document.issued_date == date(2011, 3, 3)
    assert any(node.label == "5" for node in document.nodes)
    assert any(node.label == "(4)" for node in document.nodes)


def test_parse_go_order_extracts_statement_table_for_ms39() -> None:
    from stresslab.ingest import parse_go_order

    document = parse_go_order(Path("tests/Data/2011SE_MS39.PDF"))

    assert document.doc_id == "2011SE_MS39"
    assert "Adult Education Centre" in document.title
    assert document.department == "SCHOOL EDUCATION (PE.PROG.1) DEPARTMENT"
    assert document.go_number == "G.O.Ms.No. 39"
    assert document.issued_date == date(2011, 3, 25)
    assert any(table.title and "Statement" in table.title for table in document.tables)


def test_extract_nodes_ignores_trailer_lines_after_body() -> None:
    parse_module = importlib.import_module("stresslab.ingest.parse_go_order")

    lines = [
        parse_module._PageLine(page=1, text="ORDER"),
        parse_module._PageLine(page=1, text="1. Operative clause."),
        parse_module._PageLine(page=1, text="//FORWARDED BY ORDER//"),
        parse_module._PageLine(page=1, text="SECTION OFFICER"),
    ]

    nodes = parse_module._extract_nodes(lines)

    assert len(nodes) == 1
    assert nodes[0].label == "1"
    assert nodes[0].text == "Operative clause."


def test_extract_tables_stops_short_statement_when_body_resumes() -> None:
    parse_module = importlib.import_module("stresslab.ingest.parse_go_order")

    lines = [
        parse_module._PageLine(page=1, text="Statement - A"),
        parse_module._PageLine(page=1, text="District Posts"),
        parse_module._PageLine(page=1, text="3. Body text resumes here."),
        parse_module._PageLine(page=1, text="Further narrative line."),
    ]

    tables = parse_module._extract_tables(lines)

    assert len(tables) == 1
    assert tables[0].title == "Statement - A"
    assert tables[0].rows == [["District Posts"]]
