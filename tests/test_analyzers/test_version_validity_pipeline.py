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


def run_with_multiple_docs(
    *,
    query: str,
    final_answer: str,
    chunks: list[RetrievedChunk],
    cited_doc_ids: list[str],
    corpus_entries: list[CorpusEntry],
) -> RAGRun:
    return RAGRun(
        run_id="run-version-pipeline-multi",
        query=query,
        retrieved_chunks=chunks,
        final_answer=final_answer,
        cited_doc_ids=cited_doc_ids,
        corpus_entries=corpus_entries,
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


def test_version_validity_can_select_stale_retrieval_for_invalid_source() -> None:
    diagnosis = diagnose(
        run_with_doc(metadata={"status": "withdrawn", "withdrawn_by_doc_ids": ["doc-2"]})
    )

    assert diagnosis.version_validity_report is not None
    assert "doc-1" in diagnosis.version_validity_report.withdrawn_doc_ids
    assert diagnosis.primary_failure.name == "STALE_RETRIEVAL"


def test_expired_cited_source_outranks_unsupported_claim() -> None:
    diagnosis = diagnose(
        run_with_multiple_docs(
            query="What is the current policy?",
            final_answer="Policy X is the current policy [doc-expired].",
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-expired",
                    text="Policy X (Expired 2021)",
                    source_doc_id="doc-expired",
                    score=0.9,
                    metadata={},
                )
            ],
            cited_doc_ids=["doc-expired"],
            corpus_entries=[
                CorpusEntry(
                    doc_id="doc-expired",
                    text="Policy X (Expired 2021)",
                    timestamp=datetime(2021, 1, 1, tzinfo=UTC),
                    metadata={"expiry_date": "2021-12-31"},
                )
            ],
        )
    )

    assert diagnosis.primary_failure.name == "STALE_RETRIEVAL"
    assert diagnosis.root_cause_stage.name == "RETRIEVAL"


def test_withdrawn_cited_source_outranks_unsupported_claim() -> None:
    diagnosis = diagnose(
        run_with_multiple_docs(
            query="Can I use tool A?",
            final_answer="Yes, use Tool A [doc-withdrawn].",
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-withdrawn",
                    text="Tool A (Withdrawn due to safety)",
                    source_doc_id="doc-withdrawn",
                    score=0.9,
                    metadata={},
                )
            ],
            cited_doc_ids=["doc-withdrawn"],
            corpus_entries=[
                CorpusEntry(
                    doc_id="doc-withdrawn",
                    text="Tool A (Withdrawn due to safety)",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                    metadata={"status": "withdrawn"},
                )
            ],
        )
    )

    assert diagnosis.primary_failure.name == "STALE_RETRIEVAL"
    assert diagnosis.root_cause_stage.name == "RETRIEVAL"


def test_stale_retrieved_only_blocks_clean_when_retrieval_quality_affected() -> None:
    diagnosis = diagnose(
        run_with_multiple_docs(
            query="Who is the CEO?",
            final_answer="The CEO is Alice [doc-new].",
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
            corpus_entries=[
                CorpusEntry(
                    doc_id="doc-old",
                    text="Old CEO Bob (2010)",
                    timestamp=datetime(2010, 1, 1, tzinfo=UTC),
                    metadata={},
                ),
                CorpusEntry(
                    doc_id="doc-new",
                    text="New CEO Alice (2024)",
                    timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    metadata={"status": "active"},
                ),
            ],
        )
    )

    assert diagnosis.primary_failure.name == "STALE_RETRIEVAL"
    assert diagnosis.root_cause_stage.name == "RETRIEVAL"


def test_stale_irrelevant_source_does_not_primary_fail() -> None:
    diagnosis = diagnose(
        run_with_multiple_docs(
            query="Who is the CEO?",
            final_answer="The CEO is Alice [doc-ceo].",
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-lease",
                    text="Legacy office lease terms (2010)",
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
            corpus_entries=[
                CorpusEntry(
                    doc_id="doc-lease",
                    text="Legacy office lease terms (2010)",
                    timestamp=datetime(2010, 1, 1, tzinfo=UTC),
                    metadata={},
                ),
                CorpusEntry(
                    doc_id="doc-ceo",
                    text="New CEO Alice (2024)",
                    timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    metadata={"status": "active"},
                ),
            ],
        )
    )

    assert diagnosis.primary_failure.name != "STALE_RETRIEVAL"


def test_version_validity_decision_trace_explains_downstream_claim_failure() -> None:
    diagnosis = diagnose(
        run_with_multiple_docs(
            query="What is the current policy?",
            final_answer="Policy X is the current policy [doc-expired].",
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-expired",
                    text="Policy X (Expired 2021)",
                    source_doc_id="doc-expired",
                    score=0.9,
                    metadata={},
                )
            ],
            cited_doc_ids=["doc-expired"],
            corpus_entries=[
                CorpusEntry(
                    doc_id="doc-expired",
                    text="Policy X (Expired 2021)",
                    timestamp=datetime(2021, 1, 1, tzinfo=UTC),
                    metadata={"expiry_date": "2021-12-31"},
                )
            ],
        )
    )

    assert diagnosis.diagnosis_decision_trace is not None
    assert "downstream of invalid source lifecycle evidence" in diagnosis.diagnosis_decision_trace["selection_reason"]
