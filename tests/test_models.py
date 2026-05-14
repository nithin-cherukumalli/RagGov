"""Tests for core RagGov Pydantic models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    Diagnosis,
    FailureStage,
    FailureType,
    SecurityRisk,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


def test_retrieved_chunk_has_word_count_and_forbids_extra_fields() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="alpha beta gamma",
        source_doc_id="doc-1",
        score=0.82,
    )

    assert chunk.word_count == 3
    assert chunk.metadata == {}

    with pytest.raises(ValidationError):
        RetrievedChunk(
            chunk_id="chunk-1",
            text="alpha",
            source_doc_id="doc-1",
            score=None,
            unexpected=True,  # type: ignore[call-arg]
        )


def test_corpus_entry_defaults_and_metadata_are_not_shared() -> None:
    first = CorpusEntry(doc_id="doc-1", text="one", timestamp=None)
    second = CorpusEntry(doc_id="doc-2", text="two", timestamp=None)

    first.metadata["source"] = "fixture"

    assert second.metadata == {}


def test_rag_run_defaults_and_chunk_helpers() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="first chunk",
        source_doc_id="doc-1",
        score=None,
    )
    other_chunk = RetrievedChunk(
        chunk_id="chunk-2",
        text="second chunk",
        source_doc_id="doc-2",
        score=0.75,
    )
    corpus_entry = CorpusEntry(
        doc_id="doc-1",
        text="source text",
        timestamp=datetime(2026, 4, 10, tzinfo=UTC),
    )

    run = RAGRun(
        query="What happened?",
        retrieved_chunks=[chunk, other_chunk],
        final_answer="An answer.",
        corpus_entries=[corpus_entry],
    )

    assert run.run_id
    assert run.created_at.tzinfo is not None
    assert run.cited_doc_ids == []
    assert run.answer_confidence is None
    assert run.trace is None
    assert run.metadata == {}
    assert run.chunk_by_id("doc-1") == chunk
    assert run.chunk_by_id("missing") is None
    assert run.all_chunk_text() == "first chunk\nsecond chunk"


def test_run_mutable_defaults_are_not_shared() -> None:
    first = RAGRun(query="q1", retrieved_chunks=[], final_answer="a1")
    second = RAGRun(query="q2", retrieved_chunks=[], final_answer="a2")

    first.cited_doc_ids.append("doc-1")
    first.corpus_entries.append(CorpusEntry(doc_id="doc-1", text="text", timestamp=None))
    first.metadata["key"] = "value"

    assert second.cited_doc_ids == []
    assert second.corpus_entries == []
    assert second.metadata == {}


def test_diagnosis_models_defaults_enums_and_summary() -> None:
    claim_result = ClaimResult(
        claim_text="The policy changed.",
        label="unsupported",
    )
    analyzer_result = AnalyzerResult(
        analyzer_name="support",
        status="warn",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        evidence=["No supporting source."],
        citation_probe_results=[{"doc_id": "doc-1", "probe": "anchor", "passed": True, "anchor_terms_found": ["alpha"]}],
    )
    diagnosis = Diagnosis(
        run_id="run-1",
        primary_failure=FailureType.UNSUPPORTED_CLAIM,
        root_cause_stage=FailureStage.GROUNDING,
        should_have_answered=False,
        security_risk=SecurityRisk.LOW,
        diagnostic_score=0.42,
        pipeline_health_score=0.6,
        first_failing_node="CLAIM_GROUNDING",
        citation_faithfulness="partial",
        claim_results=[claim_result],
        evidence=["The answer lacks support."],
        recommended_fix="Retrieve stronger evidence.",
        analyzer_results=[analyzer_result],
    )

    assert FailureStage.RETRIEVAL == "RETRIEVAL"
    assert FailureType.CLEAN == "CLEAN"
    assert FailureType.POST_RATIONALIZED_CITATION == "POST_RATIONALIZED_CITATION"
    assert FailureType.TABLE_STRUCTURE_LOSS == "TABLE_STRUCTURE_LOSS"
    assert FailureType.HIERARCHY_FLATTENING == "HIERARCHY_FLATTENING"
    assert FailureType.METADATA_LOSS == "METADATA_LOSS"
    assert SecurityRisk.HIGH == "HIGH"
    assert diagnosis.created_at.tzinfo is not None
    assert diagnosis.secondary_failures == []
    assert diagnosis.checks_run == []
    assert diagnosis.checks_skipped == []
    assert diagnosis.citation_faithfulness == "partial"
    assert diagnosis.analyzer_results[0].citation_probe_results is not None
    assert diagnosis.summary() == (
        "Run run-1 | UNSUPPORTED_CLAIM | Stage: GROUNDING\n"
        "Should answer: False | Risk: LOW | Confidence Signal (uncalibrated): 0.42 | "
        "Status: uncalibrated\n"
        "Pipeline health: 60% | First failure: CLAIM_GROUNDING\n"
        "Fix: Retrieve stronger evidence."
    )


def test_diagnosis_mutable_defaults_are_not_shared() -> None:
    first = Diagnosis(
        run_id="run-1",
        primary_failure=FailureType.CLEAN,
        root_cause_stage=FailureStage.UNKNOWN,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        recommended_fix="No fix needed.",
    )
    second = Diagnosis(
        run_id="run-2",
        primary_failure=FailureType.CLEAN,
        root_cause_stage=FailureStage.UNKNOWN,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        recommended_fix="No fix needed.",
    )

    first.secondary_failures.append(FailureType.LOW_CONFIDENCE)
    first.evidence.append("evidence")
    first.checks_run.append("confidence")
    first.checks_skipped.append("security")
    first.analyzer_results.append(
        AnalyzerResult(analyzer_name="confidence", status="skip")
    )

    assert second.secondary_failures == []
    assert second.evidence == []
    assert second.checks_run == []
    assert second.checks_skipped == []
    assert second.analyzer_results == []


def test_analyzer_result_supports_typed_claim_results() -> None:
    claim = ClaimResult(
        claim_text="Refunds are available within 14 days.",
        label="entailed",
        supporting_chunk_ids=["c1"],
        confidence=0.9,
    )
    result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="pass",
        claim_results=[claim],
    )
    assert result.claim_results is not None
    assert result.claim_results[0].label == "entailed"


def test_analyzer_result_supports_typed_sufficiency_result() -> None:
    result = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="pass",
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            missing_evidence=["Claim A"],
            affected_claims=["Claim A", "Claim B"],
            evidence_chunk_ids=["chunk-1"],
            method="heuristic_claim_aware_v0",
            calibration_status="uncalibrated",
        ),
    )

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.method == "heuristic_claim_aware_v0"
