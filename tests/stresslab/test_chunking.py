"""Tests for chunking parsed ingest documents."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stresslab.chunking import ChunkRecord, FixedChunker, HierarchicalChunker
from stresslab.ingest import parse_go_order
from stresslab.ingest.models import ParsedDocument, ParsedNode, ParsedTable


def test_fixed_chunker_preserves_lineage_for_node_chunks() -> None:
    parsed_doc = ParsedDocument(
        doc_id="doc-1",
        source_path="/tmp/doc.pdf",
        title="Sample",
        nodes=[
            ParsedNode(
                node_id="node-1",
                label="body",
                text="alpha beta gamma delta epsilon",
                page_start=2,
                page_end=3,
                parent_node_id="section-1",
                section_path=["chapter", "section"],
            )
        ],
    )

    chunks = FixedChunker(window_size=3, overlap=1).chunk(parsed_doc)

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:node-1:0",
        "doc-1:node-1:1",
    ]
    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta epsilon",
    ]
    assert all(chunk.source_doc_id == "doc-1" for chunk in chunks)
    assert all(chunk.parent_node_id == "section-1" for chunk in chunks)
    assert all(chunk.page_start == 2 for chunk in chunks)
    assert all(chunk.page_end == 3 for chunk in chunks)
    assert all(chunk.section_path == ["chapter", "section"] for chunk in chunks)
    assert all(chunk.chunk_strategy == "fixed" for chunk in chunks)


def test_chunk_record_rejects_invalid_page_range() -> None:
    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id="chunk-1",
            source_doc_id="doc-1",
            text="alpha beta",
            page_start=4,
            page_end=2,
            section_path=[],
            parent_node_id=None,
            chunk_strategy="fixed",
        )


def test_hierarchical_chunker_keeps_rule_5_4_text_with_section_lineage() -> None:
    parsed = parse_go_order(Path("tests/Data/2011SE_MS20.PDF"))

    chunks = HierarchicalChunker().chunk(parsed)

    matching_chunks = [
        chunk
        for chunk in chunks
        if "no school exists" in chunk.text.lower()
    ]

    assert matching_chunks
    assert any(chunk.section_path[:2] == ["5", "(4)"] for chunk in matching_chunks)
    assert all(chunk.chunk_strategy == "hierarchical" for chunk in matching_chunks)


def test_hierarchical_chunker_assigns_non_empty_section_path_to_table_chunks() -> None:
    parsed_doc = ParsedDocument(
        doc_id="doc-1",
        source_path="/tmp/doc.pdf",
        title="Sample",
        tables=[
            ParsedTable(
                table_id="table-1",
                page=3,
                title="Statement - A",
                headers=["District", "Posts"],
                rows=[["HQ", "2"]],
            )
        ],
    )

    chunks = HierarchicalChunker().chunk(parsed_doc)

    assert len(chunks) == 1
    assert chunks[0].section_path == ["Statement - A", "table-1"]


def test_hierarchical_chunker_handles_deep_hierarchy_without_recursive_descent() -> None:
    depth = 1100
    nodes = [
        ParsedNode(
            node_id=f"node-{index}",
            label=str(index),
            text=f"level-{index}",
            page_start=1,
            page_end=1,
            parent_node_id=None if index == 1 else f"node-{index - 1}",
            section_path=[str(index)],
        )
        for index in range(1, depth + 1)
    ]
    parsed_doc = ParsedDocument(
        doc_id="doc-1",
        source_path="/tmp/doc.pdf",
        title="Deep Sample",
        nodes=nodes,
    )

    chunks = HierarchicalChunker().chunk(parsed_doc)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    assert len(chunks) == depth
    assert "1100 level-1100" in chunks_by_id["doc-1:node-1"].text
    assert "1100 level-1100" in chunks_by_id["doc-1:node-2"].text
    assert chunks_by_id["doc-1:node-1100"].text == "1100 level-1100"
