"""Ingestion-time parser validation API."""

from __future__ import annotations

from dataclasses import dataclass, field

from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ParsedDocumentIR,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
)
from raggov.parser_validation.profile import ParserValidationProfile
from raggov.parser_validation.profile_lint import ProfileLintEngine, ProfileLintReport


@dataclass(frozen=True)
class IngestionValidationRequest:
    """Document-level parser-validation request for ingestion pipelines."""

    parsed_doc: ParsedDocumentIR
    chunks: tuple[ChunkIR, ...]
    profile: ParserValidationProfile
    config: ParserValidationConfig | None = None


@dataclass(frozen=True)
class DocumentParserQualitySummary:
    """Document-level parser/chunker quality summary."""

    document_id: str
    total_chunks: int
    element_count: int
    table_count: int
    chunks_with_page_metadata: int
    chunks_with_provenance: int
    chunks_with_section_path: int
    document_quality_score: float


@dataclass(frozen=True)
class IngestionValidationReport:
    """Document-level parser ingestion validation report."""

    document_id: str
    parser_name: str | None
    total_chunks: int
    blocking_issues: tuple[ParserFinding, ...] = field(default_factory=tuple)
    warnings: tuple[ParserFinding, ...] = field(default_factory=tuple)
    recommendations: tuple[str, ...] = field(default_factory=tuple)
    quality_summary: DocumentParserQualitySummary | None = None
    lint_report: ProfileLintReport | None = None

    @property
    def blocking_issue_count(self) -> int:
        return len(self.blocking_issues)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def passed(self) -> bool:
        return not self.blocking_issues and not (self.lint_report and self.lint_report.has_errors)


def validate_ingestion(request: IngestionValidationRequest) -> IngestionValidationReport:
    """Validate one parsed document and its full chunk set at ingestion time."""
    config = request.config or ParserValidationConfig(
        chunking_profile=request.profile.chunking_strategy
    )
    lint_report = ProfileLintEngine(
        min_metadata_coverage=config.min_metadata_coverage,
        min_provenance_coverage=config.min_provenance_coverage,
    ).lint(request.chunks, config.chunking_profile, request.profile)

    findings = ParserValidationEngine(config=config).validate(
        parsed_doc=request.parsed_doc,
        chunks=request.chunks,
    )
    blocking = tuple(finding for finding in findings if finding.severity == ParserSeverity.FAIL)
    warnings = tuple(finding for finding in findings if finding.severity == ParserSeverity.WARN)
    recommendations = tuple(
        dict.fromkeys(finding.remediation for finding in (*blocking, *warnings))
    )

    return IngestionValidationReport(
        document_id=request.parsed_doc.document_id,
        parser_name=request.parsed_doc.parser_name,
        total_chunks=len(request.chunks),
        blocking_issues=blocking,
        warnings=warnings,
        recommendations=recommendations,
        quality_summary=_quality_summary(request.parsed_doc, request.chunks, findings, lint_report),
        lint_report=lint_report,
    )


def _quality_summary(
    parsed_doc: ParsedDocumentIR,
    chunks: tuple[ChunkIR, ...],
    findings: list[ParserFinding],
    lint_report: ProfileLintReport,
) -> DocumentParserQualitySummary:
    total_chunks = len(chunks)
    chunks_with_page = sum(1 for chunk in chunks if chunk.page_start is not None)
    chunks_with_provenance = sum(
        1 for chunk in chunks if chunk.source_element_ids or chunk.source_table_ids
    )
    chunks_with_section = sum(1 for chunk in chunks if chunk.section_path)

    issue_count = len(findings) + len(lint_report.errors) + len(lint_report.warnings)
    document_quality_score = max(0.0, 1.0 - min(1.0, issue_count / max(total_chunks, 1)))

    return DocumentParserQualitySummary(
        document_id=parsed_doc.document_id,
        total_chunks=total_chunks,
        element_count=len(parsed_doc.elements),
        table_count=len(parsed_doc.tables),
        chunks_with_page_metadata=chunks_with_page,
        chunks_with_provenance=chunks_with_provenance,
        chunks_with_section_path=chunks_with_section,
        document_quality_score=document_quality_score,
    )
