from __future__ import annotations

from typing import Iterable, Sequence

from raggov.parser_validation.models import (
    ChunkIR,
    ParsedDocumentIR,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
)
from raggov.parser_validation.validators.base import ParserValidator
from raggov.parser_validation.validators.chunk_boundary import ChunkBoundaryValidator
from raggov.parser_validation.validators.hierarchy import HierarchyValidator
from raggov.parser_validation.validators.metadata import MetadataValidator
from raggov.parser_validation.validators.table_structure import TableStructureValidator


class ParserValidationEngine:
    """Runs parser-stage structural validators over parsed-document and chunk IR."""

    def __init__(
        self,
        validators: Iterable[ParserValidator] | None = None,
        config: ParserValidationConfig | None = None,
    ) -> None:
        self.config = config or ParserValidationConfig()
        if validators is None:
            validators = [
                TableStructureValidator(),
                MetadataValidator(),
                HierarchyValidator(),
                ChunkBoundaryValidator(),
            ]
        self.validators = list(validators)

    def validate(
        self,
        parsed_doc: ParsedDocumentIR | None,
        chunks: Sequence[ChunkIR],
    ) -> list[ParserFinding]:
        findings: list[ParserFinding] = []

        for validator in self.validators:
            findings.extend(validator.validate(parsed_doc, chunks, self.config))

        return sorted(
            findings,
            key=lambda finding: (
                self._severity_rank(finding.severity),
                -finding.confidence,
                finding.validator_name,
            ),
        )

    def _severity_rank(self, severity: ParserSeverity) -> int:
        if severity == ParserSeverity.FAIL:
            return 0
        if severity == ParserSeverity.WARN:
            return 1
        return 2
