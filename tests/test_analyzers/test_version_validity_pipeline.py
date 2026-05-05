"""Pipeline integration tests for TemporalSourceValidityAnalyzerV1."""

from __future__ import annotations

from datetime import UTC, datetime

from raggov.analyzers.version_validity import (
    TemporalSourceValidityAnalyzerV1,
    VersionValidityAnalyzerV1,
)
from raggov.engine import DiagnosisEngine, diagnose
from raggov.io.serialize import diagnosis_to_dict
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import (
    CitationFaithfulnessReport,
    ClaimCitationFaithfulnessRecord,
)
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun


def chunk(doc_id: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{doc_id}",
        text=f"text for {doc_id}",
        source_doc_id=doc_id,
        score=0.9,
        metadata=metadata or {},
    )


def run_with_doc(
    *,
    metadata: dict | None = None,
    timestamp: datetime | None = None,
    citation_report: CitationFaithfulnessReport | None = None,
) -> RAGRun:
    return RAGRun(
        run_id="run-version-pipeline",
        query="query",
        retrieved_chunks=[chunk("doc-1")],
        final_answer="Answer.",
        cited_doc_ids=["doc-1"],
        corpus_entries=[
            CorpusEntry(
                doc_id="doc-1",
                text="",
                timestamp=timestamp,
                metadata=metadata or {},
            )
        ],
        citation_faithfulness_report=citation_report,
        metadata={"query_date": "2026-05-02T00:00:00+00:00"},
    )


def test_version_validity_analyzer_is_importable() -> None:
    assert TemporalSourceValidityAnalyzerV1.__name__ == "TemporalSourceValidityAnalyzerV1"
    assert VersionValidityAnalyzerV1.__name__ == "VersionValidityAnalyzerV1"


def test_default_pipeline_order_places_version_validity_after_citation_faithfulness() -> None:
    engine = DiagnosisEngine(config={})
    names = [analyzer.__class__.__name__ for analyzer in engine.analyzers]

    assert "TemporalSourceValidityAnalyzerV1" in names
    assert names.index("CitationMismatchAnalyzer") < names.index("CitationFaithfulnessAnalyzerV0")
    assert names.index("CitationFaithfulnessAnalyzerV0") < names.index("TemporalSourceValidityAnalyzerV1")


def test_default_diagnosis_includes_version_validity_section() -> None:
    diagnosis = diagnose(run_with_doc(metadata={"status": "active"}))
    payload = diagnosis_to_dict(diagnosis)

    assert "TemporalSourceValidityAnalyzerV1" in payload["checks_run"]
    section = payload["version_validity_report"]
    assert section["query_date"] == "2026-05-02T00:00:00Z"
    assert section["method_type"] == "practical_approximation"
    assert section["calibration_status"] == "uncalibrated"
    assert section["recommended_for_gating"] is False
    assert section["limitations"]
    assert section["active_doc_ids"] == ["doc-1"]
    assert section["stale_doc_ids"] == []
    assert section["superseded_doc_ids"] == []
    assert section["amended_doc_ids"] == []
    assert section["withdrawn_doc_ids"] == []
    assert section["replaced_doc_ids"] == []
    assert section["deprecated_doc_ids"] == []
    assert section["expired_doc_ids"] == []
    assert section["not_yet_effective_doc_ids"] == []
    assert section["metadata_missing_doc_ids"] == []
    assert section["high_risk_claim_ids"] == []
    record = section["document_records"][0]
    assert record["doc_id"] == "doc-1"
    assert record["validity_status"] == "active"
    assert record["validity_risk"] == "low"
    assert record["evidence_source"] == "corpus_metadata"
    assert record["evidence_paths"] == ["metadata.status"]
    assert "explanation" in record
    assert "warnings" in record


def test_report_clearly_marks_missing_metadata() -> None:
    diagnosis = diagnose(run_with_doc())
    section = diagnosis_to_dict(diagnosis)["version_validity_report"]

    assert section["metadata_missing_doc_ids"] == ["doc-1"]
    assert section["document_records"][0]["validity_status"] == "metadata_missing"
    assert section["lineage_metadata_used"] is False
    assert section["age_based_fallback_used"] is False


def test_report_clearly_marks_age_based_fallback_type() -> None:
    old_timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    diagnosis = diagnose(run_with_doc(timestamp=old_timestamp))
    section = diagnosis_to_dict(diagnosis)["version_validity_report"]

    assert section["stale_doc_ids"] == ["doc-1"]
    assert section["age_based_fallback_used"] is True
    assert section["document_records"][0]["evidence_source"] == "heuristic_age_check"
    assert (
        "age-based staleness is only a heuristic freshness warning"
        in section["document_records"][0]["warnings"]
    )


def test_dependency_visibility_marks_retrieval_and_citation_reports() -> None:
    citation_report = CitationFaithfulnessReport(
        records=[
            ClaimCitationFaithfulnessRecord(
                claim_id="claim-1",
                claim_text="A claim.",
                cited_doc_ids=["doc-1"],
            )
        ]
    )
    diagnosis = diagnose(
        run_with_doc(
            metadata={"status": "superseded", "superseded_by_doc_ids": ["doc-2"]},
            citation_report=citation_report,
        )
    )
    section = diagnosis_to_dict(diagnosis)["version_validity_report"]

    assert section["retrieval_evidence_profile_used"] is True
    assert section["citation_faithfulness_report_used"] is True
    assert section["lineage_metadata_used"] is True
    assert section["age_based_fallback_used"] is False
    claim = section["claim_source_records"][0]
    assert claim["claim_id"] == "claim-1"
    assert claim["cited_doc_ids"] == ["doc-1"]
    assert claim["invalid_cited_doc_ids"] == ["doc-1"]
    assert claim["claim_validity_status"] == "superseded"
    assert claim["claim_validity_risk"] == "high"
    assert claim["does_invalid_source_affect_claim"] is None
    assert "explanation" in claim


def test_version_validity_does_not_block_primary_diagnosis() -> None:
    diagnosis = diagnose(
        run_with_doc(metadata={"status": "withdrawn", "withdrawn_by_doc_ids": ["doc-2"]})
    )

    assert diagnosis.version_validity_report is not None
    assert "doc-1" in diagnosis.version_validity_report.withdrawn_doc_ids
    assert diagnosis.primary_failure.name != "STALE_RETRIEVAL"
