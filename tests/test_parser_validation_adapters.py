from dataclasses import dataclass

from raggov.parser_validation.adapters import (
    chunk_from_rag_chunk,
    chunks_from_rag_run,
    parsed_doc_from_run_metadata,
)
from raggov.parser_validation.models import ParsedDocumentIR


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    metadata: dict


@dataclass
class FakeRun:
    retrieved_chunks: list
    metadata: dict | None = None


def test_chunk_from_object_with_metadata():
    chunk = FakeChunk(
        chunk_id="c1",
        text="hello",
        metadata={
            "source_element_ids": ["e1", "e2"],
            "source_table_ids": ["t1"],
            "page_start": "3",
            "page_end": "4",
            "section_path": ["Chapter 1", "Rule 2"],
        },
    )

    out = chunk_from_rag_chunk(chunk)

    assert out.chunk_id == "c1"
    assert out.text == "hello"
    assert out.source_element_ids == ("e1", "e2")
    assert out.source_table_ids == ("t1",)
    assert out.page_start == 3
    assert out.page_end == 4
    assert out.section_path == ("Chapter 1", "Rule 2")


def test_chunk_from_dict_page_content_style():
    chunk = {
        "id": "dict1",
        "page_content": "hello from page content",
        "metadata": {
            "page_number": 7,
            "table_id": "table_9",
            "headers": ["Part I", "Section 4"],
        },
    }

    out = chunk_from_rag_chunk(chunk)

    assert out.chunk_id == "dict1"
    assert out.text == "hello from page content"
    assert out.page_start == 7
    assert out.page_end == 7
    assert out.source_table_ids == ("table_9",)
    assert out.section_path == ("Part I", "Section 4")


def test_chunk_handles_missing_metadata():
    chunk = {"text": "plain text"}

    out = chunk_from_rag_chunk(chunk)

    assert out.chunk_id == "unknown_chunk"
    assert out.text == "plain text"
    assert out.metadata == {}
    assert out.source_element_ids == ()
    assert out.source_table_ids == ()


def test_chunks_from_rag_run():
    run = FakeRun(
        retrieved_chunks=[
            FakeChunk(chunk_id="c1", text="one", metadata={}),
            FakeChunk(chunk_id="c2", text="two", metadata={}),
        ]
    )

    out = chunks_from_rag_run(run)

    assert [chunk.chunk_id for chunk in out] == ["c1", "c2"]


def test_parsed_doc_from_run_metadata_direct():
    parsed_doc = ParsedDocumentIR(document_id="doc1")

    class Run:
        parsed_document_ir = parsed_doc

    assert parsed_doc_from_run_metadata(Run()) is parsed_doc


def test_parsed_doc_from_run_metadata_dict():
    parsed_doc = ParsedDocumentIR(document_id="doc1")
    run = FakeRun(retrieved_chunks=[], metadata={"parsed_document_ir": parsed_doc})

    assert parsed_doc_from_run_metadata(run) is parsed_doc
