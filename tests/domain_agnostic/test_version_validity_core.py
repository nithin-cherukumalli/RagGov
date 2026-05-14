from __future__ import annotations

from raggov.analyzers.version_validity.analyzer import VersionValidityAnalyzerV1
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import FailureType
from raggov.models.run import RAGRun
from raggov.models.version_validity import DocumentValidityStatus


def run_for(metadata: dict, text: str = "Source text.") -> RAGRun:
    return RAGRun(
        query="What applies now?",
        final_answer="Answer.",
        retrieved_chunks=[
            RetrievedChunk(chunk_id="c1", text=text, source_doc_id="doc-1", score=None)
        ],
        corpus_entries=[
            CorpusEntry(
                doc_id="doc-1",
                text=text,
                timestamp=None,
                metadata=metadata,
            )
        ],
        cited_doc_ids=["doc-1"],
        metadata={"query_date": "2026-05-13T00:00:00Z"},
    )


def test_software_deprecated_api() -> None:
    result = VersionValidityAnalyzerV1().analyze(
        run_for({"status": "deprecated", "deprecated_by": "api-v3", "version": "2.0"})
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.STALE_RETRIEVAL
    assert result.version_validity_report.document_records[0].validity_status == DocumentValidityStatus.DEPRECATED
    assert result.version_validity_report.document_records[0].version_id == "2.0"


def test_healthcare_outdated_guideline() -> None:
    result = VersionValidityAnalyzerV1({"max_age_days": 365}).analyze(
        run_for({"publication_date": "2020-01-01"})
    )

    assert result.status == "warn"
    assert result.version_validity_report.document_records[0].validity_status == DocumentValidityStatus.STALE_BY_AGE


def test_finance_expired_disclosure() -> None:
    result = VersionValidityAnalyzerV1().analyze(
        run_for({"expiry_date": "2025-12-31", "current_version": "2025.1"})
    )

    assert result.status == "fail"
    assert result.version_validity_report.document_records[0].validity_status == DocumentValidityStatus.EXPIRED
    assert result.version_validity_report.document_records[0].version_id == "2025.1"


def test_product_manual_old_version_by_age() -> None:
    result = VersionValidityAnalyzerV1({"max_age_days": 365}).analyze(
        run_for({"updated_at": "2023-01-01", "version": "1.0"})
    )

    assert result.status == "warn"
    assert result.version_validity_report.document_records[0].validity_status == DocumentValidityStatus.STALE_BY_AGE


def test_government_withdrawn_circular_uses_generic_status() -> None:
    result = VersionValidityAnalyzerV1().analyze(
        run_for({"status": "withdrawn", "withdrawn_at": "2024-01-01"})
    )

    assert result.status == "fail"
    assert result.version_validity_report.document_records[0].validity_status == DocumentValidityStatus.WITHDRAWN
