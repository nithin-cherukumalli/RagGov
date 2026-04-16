"""Tests for stresslab runners."""

from __future__ import annotations

import json
from pathlib import Path

from stresslab.chunking import FixedChunker
from stresslab.ingest.models import ParsedDocument, ParsedNode
from stresslab.runners import run_build_index, run_ingest


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(char) for char in text) % 97)
            vectors.append([length, checksum, 1.0])
        return vectors


def test_run_ingest_writes_parsed_artifact(tmp_path: Path) -> None:
    result = run_ingest(
        source_path=Path("tests/Data/2011SE_MS1.PDF"),
        output_dir=tmp_path,
    )

    assert result.parsed_path.exists()
    payload = json.loads(result.parsed_path.read_text(encoding="utf-8"))
    assert payload["doc_id"] == "2011SE_MS1"
    assert payload["source_path"].endswith("2011SE_MS1.PDF")


def test_run_build_index_writes_index_and_manifest(tmp_path: Path) -> None:
    parsed_document = ParsedDocument(
        doc_id="doc-1",
        source_path="/tmp/doc-1.pdf",
        title="Order",
        nodes=[
            ParsedNode(
                node_id="node-1",
                label="1",
                text="alpha beta gamma delta epsilon",
                page_start=1,
                page_end=1,
                section_path=["1"],
            ),
            ParsedNode(
                node_id="node-2",
                label="2",
                text="zeta eta theta iota kappa",
                page_start=2,
                page_end=2,
                section_path=["2"],
            ),
        ],
    )

    result = run_build_index(
        parsed_document=parsed_document,
        output_dir=tmp_path,
        chunker=FixedChunker(window_size=3, overlap=1),
        embedding_client=_FakeEmbeddingClient(),
    )

    assert result.index_path.exists()
    assert result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["doc_id"] == "doc-1"
    assert manifest["chunk_count"] == 4
    assert manifest["embedding_count"] == 4
    assert manifest["index_path"].endswith(".npz")
