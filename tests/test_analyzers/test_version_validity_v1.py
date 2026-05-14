"""Tests for TemporalSourceValidityAnalyzerV1."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from raggov.analyzers.version_validity import (
    TemporalSourceValidityAnalyzerV1,
    VersionValidityAnalyzerV1,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import (
    CitationFaithfulnessReport,
    ClaimCitationFaithfulnessRecord,
)
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun
from raggov.models.version_validity import (
    DocumentValidityRisk,
    DocumentValidityStatus,
    VersionValidityCalibrationStatus,
    VersionValidityMethodType,
)


QUERY_DATE = datetime(2026, 5, 2, tzinfo=UTC)


def chunk(doc_id: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{doc_id}",
        text=f"text for {doc_id}",
        source_doc_id=doc_id,
        score=None,
        metadata=metadata or {},
    )


def corpus_entry(
    doc_id: str,
    metadata: dict | None = None,
    timestamp: datetime | None = None,
) -> CorpusEntry:
    return CorpusEntry(
        doc_id=doc_id,
        text="",
        timestamp=timestamp,
        metadata=metadata or {},
    )


def run(
    *,
    chunks: list[RetrievedChunk] | None = None,
    cited_doc_ids: list[str] | None = None,
    corpus_entries: list[CorpusEntry] | None = None,
    citation_report: CitationFaithfulnessReport | None = None,
    query_date: datetime | None = QUERY_DATE,
) -> RAGRun:
    metadata = {}
    if query_date is not None:
        metadata["query_date"] = query_date.isoformat()
    return RAGRun(
        run_id="run-version",
        query="query",
        retrieved_chunks=chunks or [],
        final_answer="Answer.",
        cited_doc_ids=cited_doc_ids or [],
        corpus_entries=corpus_entries or [],
        citation_faithfulness_report=citation_report,
        metadata=metadata,
    )


def analyze(test_run: RAGRun):
    result = TemporalSourceValidityAnalyzerV1({"max_age_days": 180}).analyze(test_run)
    assert result.version_validity_report is not None
    return result, result.version_validity_report


def test_skips_when_no_retrievable_or_citable_source_exists() -> None:
    result = VersionValidityAnalyzerV1().analyze(run())

    assert result.status == "skip"
    assert result.evidence == ["no retrieved chunks or cited documents available"]
    assert result.version_validity_report is None


def test_active_document_passes() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"status": "active"})],
        )
    )

    assert result.status == "pass"
    assert report.document_records[0].validity_status == DocumentValidityStatus.ACTIVE
    assert report.document_records[0].validity_risk == DocumentValidityRisk.LOW
    assert report.active_doc_ids == ["doc-1"]


def test_superseded_document_fails() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"superseded_by_doc_ids": ["doc-2"]})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.SUPERSEDED
    assert report.superseded_doc_ids == ["doc-1"]


def test_withdrawn_document_fails() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"status": "withdrawn"})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.WITHDRAWN
    assert report.withdrawn_doc_ids == ["doc-1"]


def test_expired_document_fails() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"expiry_date": "2026-01-01"})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.EXPIRED
    assert report.expired_doc_ids == ["doc-1"]


def test_not_yet_effective_document_fails() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"effective_date": "2026-06-01"})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.NOT_YET_EFFECTIVE
    assert report.not_yet_effective_doc_ids == ["doc-1"]


def test_amended_document_warns() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"amended_by_doc_ids": ["doc-a"]})],
        )
    )

    assert result.status == "warn"
    assert report.document_records[0].validity_status == DocumentValidityStatus.AMENDED
    assert report.document_records[0].validity_risk == DocumentValidityRisk.MEDIUM
    assert report.amended_doc_ids == ["doc-1"]


def test_old_document_without_lineage_warns_as_stale_by_age() -> None:
    old_timestamp = QUERY_DATE - timedelta(days=365)
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", timestamp=old_timestamp)],
        )
    )

    record = report.document_records[0]
    assert result.status == "warn"
    assert record.validity_status == DocumentValidityStatus.STALE_BY_AGE
    assert record.validity_risk == DocumentValidityRisk.UNKNOWN
    assert report.stale_doc_ids == ["doc-1"]
    assert "age-based staleness is only a heuristic freshness warning" in record.warnings
    assert "corpus_entries.timestamp" in record.evidence_paths
    assert "config.max_age_days" in record.evidence_paths


def test_missing_query_date_assumes_current_utc_and_warns() -> None:
    _, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"status": "active"})],
            query_date=None,
        )
    )

    record = report.document_records[0]
    assert report.query_date is not None
    assert "query_date not provided; assumed current UTC date" in record.warnings


def test_missing_metadata_warns_unknown() -> None:
    result, report = analyze(run(chunks=[chunk("doc-1")]))

    assert result.status == "warn"
    assert report.document_records[0].validity_status == DocumentValidityStatus.METADATA_MISSING
    assert report.document_records[0].validity_risk == DocumentValidityRisk.UNKNOWN
    assert report.document_records[0].evidence_paths == ["metadata"]
    assert report.metadata_missing_doc_ids == ["doc-1"]


def test_generic_lifecycle_metadata_superseded_by_fails_with_evidence_paths() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("policy-v1")],
            corpus_entries=[
                corpus_entry(
                    "policy-v1",
                    {
                        "version_id": "v1",
                        "superseded_by": "policy-v2",
                        "valid_from": "2025-01-01",
                        "valid_to": "2026-12-31",
                    },
                )
            ],
        )
    )

    record = report.document_records[0]
    assert result.status == "fail"
    assert record.validity_status == DocumentValidityStatus.SUPERSEDED
    assert record.validity_risk == DocumentValidityRisk.HIGH
    assert record.version_id == "v1"
    assert record.effective_date == datetime(2025, 1, 1, tzinfo=UTC)
    assert record.expiry_date == datetime(2026, 12, 31, tzinfo=UTC)
    assert record.superseded_by_doc_ids == ["policy-v2"]
    assert "metadata.superseded_by" in record.evidence_paths


def test_deprecated_source_is_high_risk_from_status_or_lineage() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("api-v1")],
            corpus_entries=[corpus_entry("api-v1", {"deprecated_by": ["api-v2"]})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.DEPRECATED
    assert report.document_records[0].validity_risk == DocumentValidityRisk.HIGH
    assert report.deprecated_doc_ids == ["api-v1"]
    assert "metadata.deprecated_by" in report.document_records[0].evidence_paths


def test_replaced_source_is_high_risk_from_generic_lineage() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("guideline-old")],
            corpus_entries=[corpus_entry("guideline-old", {"replaced_by": "guideline-new"})],
        )
    )

    assert result.status == "fail"
    assert report.document_records[0].validity_status == DocumentValidityStatus.REPLACED
    assert report.replaced_doc_ids == ["guideline-old"]
    assert "metadata.replaced_by" in report.document_records[0].evidence_paths


def test_applicability_unknown_warns_when_scope_metadata_is_unclear() -> None:
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[
                corpus_entry(
                    "doc-1",
                    {
                        "academic_year": "2024-25",
                        "applicability_scope": "2024-25 admissions only",
                        "status": "applicability_unknown",
                    },
                )
            ],
        )
    )

    assert result.status == "warn"
    assert (
        report.document_records[0].validity_status
        == DocumentValidityStatus.APPLICABILITY_UNKNOWN
    )
    assert report.document_records[0].validity_risk == DocumentValidityRisk.UNKNOWN


def test_claim_linked_to_invalid_cited_doc_becomes_high_risk() -> None:
    citation_report = CitationFaithfulnessReport(
        records=[
            ClaimCitationFaithfulnessRecord(
                claim_id="claim-1",
                claim_text="A claim.",
                cited_doc_ids=["doc-1"],
            )
        ]
    )
    result, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"status": "superseded"})],
            citation_report=citation_report,
        )
    )

    claim_record = report.claim_source_records[0]
    assert result.status == "fail"
    assert claim_record.claim_validity_risk == DocumentValidityRisk.HIGH
    assert claim_record.invalid_cited_doc_ids == ["doc-1"]
    assert claim_record.does_invalid_source_affect_claim is None
    assert "claim.cited_doc_ids" in claim_record.evidence_paths
    assert "document_records.doc-1.validity_status" in claim_record.evidence_paths
    assert report.high_risk_claim_ids == ["claim-1"]
    assert report.cited_invalid_doc_ids == ["doc-1"]


def test_retrieved_only_stale_source_is_marked_when_query_is_current() -> None:
    result, report = analyze(
        run(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-old",
                    text="Old CEO Bob (2010)",
                    source_doc_id="doc-old",
                    score=0.4,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="chunk-new",
                    text="New CEO Alice (2024)",
                    source_doc_id="doc-new",
                    score=0.9,
                    metadata={},
                ),
            ],
            cited_doc_ids=["doc-new"],
            query_date=QUERY_DATE,
        ).model_copy(update={"query": "Who is the current CEO?", "final_answer": "The CEO is Alice."})
    )

    assert result.status == "fail"
    assert report.retrieved_only_stale_doc_ids == ["doc-old"]
    assert report.retrieval_quality_affected_doc_ids == ["doc-old"]


def test_stale_irrelevant_source_is_tracked_without_primary_failure() -> None:
    result, report = analyze(
        run(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-lease",
                    text="Legacy lease terms (2010)",
                    source_doc_id="doc-lease",
                    score=0.2,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="chunk-ceo",
                    text="New CEO Alice (2024)",
                    source_doc_id="doc-ceo",
                    score=0.9,
                    metadata={},
                ),
            ],
            cited_doc_ids=["doc-ceo"],
            query_date=QUERY_DATE,
        ).model_copy(update={"query": "Who is the CEO?", "final_answer": "The CEO is Alice."})
    )

    assert report.stale_but_irrelevant_doc_ids == ["doc-lease"]
    assert report.retrieval_quality_affected_doc_ids == []


def test_recommended_for_gating_is_false_and_metadata_uncalibrated() -> None:
    _, report = analyze(
        run(
            chunks=[chunk("doc-1")],
            corpus_entries=[corpus_entry("doc-1", {"status": "active"})],
        )
    )

    assert report.method_type == VersionValidityMethodType.PRACTICAL_APPROXIMATION
    assert report.calibration_status == VersionValidityCalibrationStatus.UNCALIBRATED
    assert report.recommended_for_gating is False
    assert "not a research-faithful VersionRAG implementation" in report.limitations
