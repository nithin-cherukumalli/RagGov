"""Data models describing analysis runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile


class RAGRun(BaseModel):
    """A single retrieval-augmented generation run and its context."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    retrieved_chunks: list[RetrievedChunk]
    final_answer: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    answer_confidence: float | None = None
    trace: dict[str, Any] | None = None
    corpus_entries: list[CorpusEntry] = Field(default_factory=list)
    retrieval_evidence_profile: RetrievalEvidenceProfile | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def chunk_by_id(self, doc_id: str) -> RetrievedChunk | None:
        """Return the first retrieved chunk matching a source document ID."""
        for chunk in self.retrieved_chunks:
            if chunk.source_doc_id == doc_id:
                return chunk
        return None

    def all_chunk_text(self) -> str:
        """Return all retrieved chunk text joined by newline characters."""
        return "\n".join(chunk.text for chunk in self.retrieved_chunks)
