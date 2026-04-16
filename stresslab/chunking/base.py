"""Base chunking models and contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stresslab.ingest.models import ParsedDocument


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_doc_id: str
    text: str
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    section_path: list[str] = Field(default_factory=list)
    parent_node_id: str | None = None
    chunk_strategy: str

    @model_validator(mode="after")
    def validate_page_range(self) -> "ChunkRecord":
        if self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self


class BaseChunker(ABC):
    """Contract for chunkers that transform parsed documents into chunks."""

    @abstractmethod
    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkRecord]:
        """Return chunk records for a parsed document."""
