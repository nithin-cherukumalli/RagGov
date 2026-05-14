"""Regression tests for structured sufficiency failure detection.

Each test corresponds to one of the five golden benchmark cases that were
failing before PR 5A:
  - sufficiency_partial_12
  - sufficiency_missing_critical_13
  - sufficiency_missing_exception_14
  - sufficiency_missing_scope_15
  - sufficiency_stale_mistaken_16

These tests verify that:
1. SufficiencyAnalyzer emits the correct structured evidence marker.
2. The full DiagnosisEngine selects the right primary failure type and stage.
3. CITATION_MISMATCH does not win when explicit sufficiency evidence exists.
4. STALE_RETRIEVAL/RETRIEVAL is preferred over INSUFFICIENT_CONTEXT/RETRIEVAL
   when stale-context-mistaken-as-sufficient is detected.
"""

from __future__ import annotations

import pytest

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


def _chunk(text: str, chunk_id: str = "c1", doc_id: str = "doc1") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, source_doc_id=doc_id, score=0.9)


def _run(query: str, chunk_text: str, answer: str) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=[_chunk(chunk_text)],
        final_answer=answer,
    )


def _native_engine() -> DiagnosisEngine:
    return DiagnosisEngine(config={"mode": "native"})


# ---------------------------------------------------------------------------
# Analyzer-level tests: check structured evidence markers are emitted
# ---------------------------------------------------------------------------


def test_sufficiency_partial_emits_partial_requirement_coverage_marker() -> None:
    """SufficiencyAnalyzer detects conditional clause not addressed in context."""
    run = _run(
        query="How do I reset my password if I lost my phone?",
        chunk_text="To reset password, use your phone for 2FA.",
        answer="You need your phone for 2FA to reset your password.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert any("[sufficiency:partial_requirement_coverage]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "partial_requirement_coverage"
    assert result.sufficiency_result.recommended_fix_category == "ABSTENTION_THRESHOLD"


def test_sufficiency_missing_critical_emits_missing_critical_requirement_marker() -> None:
    """SufficiencyAnalyzer adds critical-requirement marker on safety queries with low coverage."""
    run = _run(
        query="Is this chemical safe to mix with bleach?",
        chunk_text="Chemical X is a strong acid.",
        answer="Chemical X is a strong acid.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert any("[sufficiency:missing_critical_requirement]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "missing_critical_requirement"
    assert result.sufficiency_result.recommended_fix_category == "CRITICAL_VALUE_CHECK"


def test_sufficiency_missing_exception_emits_missing_exception_marker() -> None:
    """SufficiencyAnalyzer detects universal quantifier with scoped context."""
    run = _run(
        query="Are all employees eligible for the bonus?",
        chunk_text="Full-time employees are eligible for the bonus.",
        answer="Yes, full-time employees are eligible.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert any("[sufficiency:missing_exception]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "missing_exception"
    assert result.sufficiency_result.recommended_fix_category == "COVERAGE_EXPANSION"


def test_sufficiency_missing_scope_emits_missing_scope_condition_marker() -> None:
    """SufficiencyAnalyzer detects scope-specific context for a scope-general query."""
    run = _run(
        query="What is the sales tax?",
        chunk_text="Sales tax in California is 7.25%.",
        answer="The sales tax is 7.25%.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert any("[sufficiency:missing_scope_condition]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "missing_scope_condition"
    assert result.sufficiency_result.recommended_fix_category == "SCOPE_DISAMBIGUATION"


def test_sufficiency_missing_temporal_requirement_emits_temporal_marker() -> None:
    """SufficiencyAnalyzer detects a temporal requirement with no temporal support."""
    run = _run(
        query="Which advisory fee applies after the 2026 effective date?",
        chunk_text="The advisory fee is 1.2% for managed accounts.",
        answer="The advisory fee is 1.2% for managed accounts.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert any("[sufficiency:missing_temporal_or_freshness_requirement]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "missing_temporal_or_freshness_requirement"
    assert result.sufficiency_result.recommended_fix_category == "COVERAGE_EXPANSION"


def test_sufficiency_stale_mistaken_emits_stale_context_marker() -> None:
    """SufficiencyAnalyzer detects stale chunk mistaken as sufficient for current query."""
    run = _run(
        query="What is the current CEO's name?",
        chunk_text="CEO John Smith (as of 2020)",
        answer="The CEO is John Smith.",
    )
    result = SufficiencyAnalyzer({}).analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.STALE_RETRIEVAL
    assert result.stage == FailureStage.RETRIEVAL
    assert any("[sufficiency:stale_context_mistaken_as_sufficient]" in e for e in result.evidence)
    assert any("[sufficiency:missing_temporal_or_freshness_requirement]" in e for e in result.evidence)
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.structured_failure_reason == "stale_context_mistaken_as_sufficient"
    assert result.sufficiency_result.recommended_fix_category == "FRESHNESS_FILTER"


# ---------------------------------------------------------------------------
# Engine-level tests: check decision policy selects correct primary failure
# ---------------------------------------------------------------------------


def test_sufficiency_partial_prefers_insufficient_context_stage_sufficiency() -> None:
    """End-to-end: partial coverage case resolves to INSUFFICIENT_CONTEXT/SUFFICIENCY.

    Regression: previously got CITATION_MISMATCH/GROUNDING (warn fallback winner).
    """
    engine = _native_engine()
    run = _run(
        query="How do I reset my password if I lost my phone?",
        chunk_text="To reset password, use your phone for 2FA.",
        answer="You need your phone for 2FA to reset your password.",
    )
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT
    assert diagnosis.root_cause_stage == FailureStage.SUFFICIENCY


def test_sufficiency_missing_critical_prefers_sufficiency_stage() -> None:
    """End-to-end: critical-requirement miss resolves to INSUFFICIENT_CONTEXT/SUFFICIENCY.

    Regression: previously got INSUFFICIENT_CONTEXT/RETRIEVAL because
    RetrievalDiagnosisAnalyzerV0 had higher specificity (90) than SufficiencyAnalyzer (88).
    """
    engine = _native_engine()
    run = _run(
        query="Is this chemical safe to mix with bleach?",
        chunk_text="Chemical X is a strong acid.",
        answer="Chemical X is a strong acid.",
    )
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT
    assert diagnosis.root_cause_stage == FailureStage.SUFFICIENCY


def test_sufficiency_missing_exception_not_citation_mismatch() -> None:
    """End-to-end: missing exception resolves to INSUFFICIENT_CONTEXT/SUFFICIENCY, not CITATION_MISMATCH.

    Regression: previously got CITATION_MISMATCH/GROUNDING (warn fallback winner).
    """
    engine = _native_engine()
    run = _run(
        query="Are all employees eligible for the bonus?",
        chunk_text="Full-time employees are eligible for the bonus.",
        answer="Yes, full-time employees are eligible.",
    )
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT
    assert diagnosis.root_cause_stage == FailureStage.SUFFICIENCY
    assert diagnosis.primary_failure != FailureType.CITATION_MISMATCH


def test_sufficiency_missing_scope_not_citation_mismatch() -> None:
    """End-to-end: missing scope resolves to INSUFFICIENT_CONTEXT/SUFFICIENCY, not CITATION_MISMATCH.

    Regression: previously got CITATION_MISMATCH/GROUNDING (warn fallback winner).
    """
    engine = _native_engine()
    run = _run(
        query="What is the sales tax?",
        chunk_text="Sales tax in California is 7.25%.",
        answer="The sales tax is 7.25%.",
    )
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT
    assert diagnosis.root_cause_stage == FailureStage.SUFFICIENCY
    assert diagnosis.primary_failure != FailureType.CITATION_MISMATCH


def test_sufficiency_stale_mistaken_prefers_stale_retrieval() -> None:
    """End-to-end: stale context mistaken as sufficient resolves to STALE_RETRIEVAL/RETRIEVAL.

    Regression: previously got INSUFFICIENT_CONTEXT/RETRIEVAL from RetrievalDiagnosisAnalyzerV0.
    """
    engine = _native_engine()
    run = _run(
        query="What is the current CEO's name?",
        chunk_text="CEO John Smith (as of 2020)",
        answer="The CEO is John Smith.",
    )
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.STALE_RETRIEVAL
    assert diagnosis.root_cause_stage == FailureStage.RETRIEVAL
    assert diagnosis.primary_failure != FailureType.INSUFFICIENT_CONTEXT
