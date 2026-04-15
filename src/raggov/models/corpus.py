"""Data models describing corpora and source documents."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CorpusEntry(BaseModel):
    """A source document entry available to RagGov checks."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    doc_id: str
    text: str
    timestamp: datetime | None
    metadata: dict[str, Any] = Field(default_factory=dict)
