"""Base protocol for parser-validation checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raggov.parser_validation.models import (
    ChunkIR,
    ParsedDocumentIR,
    ParserFinding,
    ParserValidationConfig,
)


class ParserValidator(Protocol):
    name: str

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
        config: ParserValidationConfig,
    ) -> list[ParserFinding]:
        """Validate parser output and retrieved chunks."""
        ...
