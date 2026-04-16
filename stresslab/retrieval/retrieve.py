"""Retrieval orchestration over embeddings and the local vector index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raggov.models.chunk import RetrievedChunk

from stresslab.index import SearchResult, VectorIndex


@dataclass(frozen=True)
class RetrievalTrace:
    query: str
    query_vector: list[float]
    top_k: int
    results: list[SearchResult]


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    trace: RetrievalTrace


class RetrievalService:
    def __init__(self, embedding_client: Any, index: VectorIndex) -> None:
        self._embedding_client = embedding_client
        self._index = index

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        query_vector = self._embed_query(query)
        results = self._index.search(query_vector, top_k=top_k)
        return RetrievalResult(
            chunks=[self._to_retrieved_chunk(result) for result in results],
            trace=RetrievalTrace(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                results=results,
            ),
        )

    def _embed_query(self, query: str) -> list[float]:
        vectors = self._embedding_client.embed_texts([query])
        if not vectors:
            raise RuntimeError("Embedding client returned no query vector")
        return [float(value) for value in vectors[0]]

    def _to_retrieved_chunk(self, result: SearchResult) -> RetrievedChunk:
        payload = result.payload
        if not isinstance(payload, dict):
            raise RuntimeError("Index payload must be a dictionary")

        text = payload.get("text")
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"Index payload for chunk {result.chunk_id} is missing text")

        source_doc_id = payload.get("source_doc_id")
        if not isinstance(source_doc_id, str) or not source_doc_id:
            source_doc_id = result.chunk_id

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"text", "source_doc_id"}
        }

        return RetrievedChunk(
            chunk_id=result.chunk_id,
            text=text,
            source_doc_id=source_doc_id,
            score=result.score,
            metadata=metadata,
        )
