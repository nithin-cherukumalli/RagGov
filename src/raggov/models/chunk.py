"""Data models describing retrieved and analyzed corpus chunks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """A retrieved text chunk and its retrieval metadata."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    chunk_id: str
    text: str
    source_doc_id: str
    score: float | None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Return the number of whitespace-delimited words in the chunk text."""
        return len(self.text.split())
