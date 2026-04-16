"""Runner for building and persisting a local vector index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from stresslab.chunking import BaseChunker
from stresslab.index import VectorIndex
from stresslab.ingest.models import ParsedDocument
from stresslab.reports import write_json_artifact


class EmbeddingClientLike(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class BuildIndexRunResult:
    index_path: Path
    manifest_path: Path
    chunk_count: int
    embedding_count: int


def run_build_index(
    parsed_document: ParsedDocument,
    output_dir: str | Path,
    chunker: BaseChunker,
    embedding_client: EmbeddingClientLike,
) -> BuildIndexRunResult:
    output = Path(output_dir)
    chunks = chunker.chunk(parsed_document)
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}"
        )

    index = VectorIndex()
    for chunk, vector in zip(chunks, embeddings, strict=True):
        index.add(chunk.chunk_id, vector, chunk.model_dump(mode="json"))

    index_path = output / f"{parsed_document.doc_id}.index.npz"
    index.save(index_path)

    manifest_path = output / f"{parsed_document.doc_id}.index-manifest.json"
    write_json_artifact(
        manifest_path,
        {
            "doc_id": parsed_document.doc_id,
            "index_path": str(index_path),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "chunker": chunker.__class__.__name__,
            "source_path": parsed_document.source_path,
        },
    )

    return BuildIndexRunResult(
        index_path=index_path,
        manifest_path=manifest_path,
        chunk_count=len(chunks),
        embedding_count=len(embeddings),
    )
