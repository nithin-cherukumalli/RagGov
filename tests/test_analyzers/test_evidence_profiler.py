"""
Tests for RetrievalEvidenceProfilerV0.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from raggov.analyzers.retrieval.evidence_profile import (
    RetrievalEvidenceProfilerV0,
    _DEFAULT_MAX_AGE_DAYS,
    _DEFAULT_MIN_OVERLAP_RATIO,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import (
    CalibrationStatus,
    CitationStatus,
    FreshnessStatus,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalMethodType,
)
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def chunk(chunk_id: str, text: str, source_doc_id: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id or chunk_id,
        score=None,
    )


def corpus_entry(doc_id: str, *, age_days: int) -> CorpusEntry:
    return CorpusEntry(
        doc_id=doc_id,
        text="",
        timestamp=datetime.now(UTC) - timedelta(days=age_days),
    )


def run(
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is the refund policy?",
    cited_doc_ids: list[str] | None = None,
    corpus_entries: list[CorpusEntry] | None = None,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer="Answer.",
        cited_doc_ids=cited_doc_ids or [],
        corpus_entries=corpus_entries or [],
    )


# ---------------------------------------------------------------------------
# Default constant alignment tests
# ---------------------------------------------------------------------------

def test_default_min_overlap_ratio_matches_scope_violation_analyzer():
    assert _DEFAULT_MIN_OVERLAP_RATIO == 0.1


def test_default_max_age_days_matches_stale_retrieval_analyzer():
    assert _DEFAULT_MAX_AGE_DAYS == 180


# ---------------------------------------------------------------------------
# Required metadata invariants
# ---------------------------------------------------------------------------

def test_profile_is_heuristic_baseline():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([chunk("c1", "refund policy covers returns")]))
    assert profile.method_type == RetrievalMethodType.HEURISTIC_BASELINE


def test_profile_is_uncalibrated():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([chunk("c1", "refund policy covers returns")]))
    assert profile.calibration_status == CalibrationStatus.UNCALIBRATED


def test_recommended_for_gating_is_false():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([chunk("c1", "refund policy covers returns")]))
    assert profile.recommended_for_gating is False


def test_limitations_present_on_normal_run():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([chunk("c1", "refund policy covers returns")]))
    assert "v0 uses lexical overlap, not semantic relevance" in profile.limitations
    assert "v0 contradiction detection uses negation heuristics, not NLI" in profile.limitations
    assert "v0 freshness uses age threshold, not legal/version validity" in profile.limitations
    assert "v0 citation detection checks provenance IDs only, not claim-level support" in profile.limitations


# ---------------------------------------------------------------------------
# No retrieved chunks
# ---------------------------------------------------------------------------

def test_no_chunks_returns_safe_profile():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([]))
    assert profile.chunks == []
    assert profile.method_type == RetrievalMethodType.HEURISTIC_BASELINE
    assert profile.calibration_status == CalibrationStatus.UNCALIBRATED
    assert profile.recommended_for_gating is False
    assert "no retrieved chunks available" in profile.limitations


def test_no_chunks_profile_does_not_include_full_limitations():
    profiler = RetrievalEvidenceProfilerV0()
    profile = profiler.build(run([]))
    assert profile.limitations == ["no retrieved chunks available"]


# ---------------------------------------------------------------------------
# Phantom citation detection
# ---------------------------------------------------------------------------

def test_detects_phantom_citation():
    r = run(
        [chunk("c1", "refund policy text", "doc-1")],
        cited_doc_ids=["doc-1", "doc-phantom"],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert "doc-phantom" in profile.phantom_citation_doc_ids
    assert "doc-1" not in profile.phantom_citation_doc_ids


def test_no_phantom_citations_when_all_cited_docs_retrieved():
    r = run(
        [chunk("c1", "refund policy text", "doc-1")],
        cited_doc_ids=["doc-1"],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.phantom_citation_doc_ids == []


def test_phantom_citation_reflected_in_degraded_status():
    r = run(
        [chunk("c1", "refund policy", "doc-1")],
        cited_doc_ids=["doc-ghost"],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.overall_retrieval_status == "degraded"


# ---------------------------------------------------------------------------
# Lexical relevance detection
# ---------------------------------------------------------------------------

def test_detects_low_lexical_relevance():
    r = run(
        [chunk("c1", "kernel driver firmware patch")],
        query="refund policy returns",
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    c = profile.chunks[0]
    assert c.query_relevance_label == QueryRelevanceLabel.IRRELEVANT
    assert c.query_relevance_score == pytest.approx(0.0)
    assert c.relevance_method == RelevanceMethod.LEXICAL_OVERLAP
    assert "c1" in profile.noisy_chunk_ids


def test_relevant_chunk_not_in_noisy_ids():
    r = run(
        [chunk("c1", "The refund policy covers all returns and credits.")],
        query="refund policy returns",
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].query_relevance_label == QueryRelevanceLabel.RELEVANT
    assert "c1" not in profile.noisy_chunk_ids


def test_partial_relevance_chunk_in_noisy_ids():
    # overlap_ratio > 0 but < threshold (0.1)
    # query terms: {refund, policy, returns} = 3 terms
    # chunk has "refund" → overlap=1/3=0.33 which is > 0.1 → RELEVANT at default threshold
    # Use a higher threshold via config to force partial
    r = run(
        [chunk("c1", "refund coverage details")],
        query="refund policy returns",
    )
    profile = RetrievalEvidenceProfilerV0({"min_overlap_ratio": 0.5}).build(r)
    c = profile.chunks[0]
    assert c.query_relevance_label == QueryRelevanceLabel.PARTIAL
    assert "c1" in profile.noisy_chunk_ids


def test_unknown_relevance_when_query_has_only_stopwords():
    r = run(
        [chunk("c1", "refund policy")],
        query="and or the",
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    c = profile.chunks[0]
    assert c.query_relevance_label == QueryRelevanceLabel.UNKNOWN
    assert c.query_relevance_score is None
    assert "c1" not in profile.noisy_chunk_ids


# ---------------------------------------------------------------------------
# Stale retrieval detection
# ---------------------------------------------------------------------------

def test_detects_stale_by_age():
    old_entry = corpus_entry("doc-old", age_days=200)
    r = run(
        [chunk("c1", "old document text", "doc-old")],
        corpus_entries=[old_entry],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].freshness_status == FreshnessStatus.STALE_BY_AGE
    assert "doc-old" in profile.stale_doc_ids


def test_fresh_document_not_stale():
    fresh_entry = corpus_entry("doc-fresh", age_days=10)
    r = run(
        [chunk("c1", "fresh content", "doc-fresh")],
        corpus_entries=[fresh_entry],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].freshness_status == FreshnessStatus.VALID
    assert profile.stale_doc_ids == []


def test_stale_detection_uses_configurable_threshold():
    entry = corpus_entry("doc-mid", age_days=45)
    r = run(
        [chunk("c1", "mid-age document", "doc-mid")],
        corpus_entries=[entry],
    )
    # Default 180 days: not stale
    profile_default = RetrievalEvidenceProfilerV0().build(r)
    assert profile_default.chunks[0].freshness_status == FreshnessStatus.VALID

    # Custom 30 days: stale
    profile_custom = RetrievalEvidenceProfilerV0({"max_age_days": 30}).build(r)
    assert profile_custom.chunks[0].freshness_status == FreshnessStatus.STALE_BY_AGE


def test_unknown_freshness_when_no_corpus_entry():
    r = run([chunk("c1", "text without corpus entry", "doc-unknown")])
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].freshness_status == FreshnessStatus.UNKNOWN
    assert profile.chunks[0].warnings == []


def test_stale_doc_ids_deduplicated():
    old_entry = corpus_entry("doc-old", age_days=200)
    r = run(
        [
            chunk("c1", "old chunk one", "doc-old"),
            chunk("c2", "old chunk two", "doc-old"),
        ],
        corpus_entries=[old_entry],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.stale_doc_ids.count("doc-old") == 1


# ---------------------------------------------------------------------------
# Contradiction candidate detection
# ---------------------------------------------------------------------------

def test_detects_negation_contradiction_candidate():
    r = run([
        chunk("c1", "The refund policy applies to hardware returns."),
        chunk("c2", "Refund policy does not apply to hardware returns."),
    ])
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert ("c1", "c2") in profile.contradictory_pairs


def test_no_contradiction_when_chunks_agree():
    r = run([
        chunk("c1", "Refunds are available within 30 days."),
        chunk("c2", "Customers may request a refund within 30 days."),
    ])
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.contradictory_pairs == []


# ---------------------------------------------------------------------------
# Citation status per chunk
# ---------------------------------------------------------------------------

def test_chunk_citation_status_cited():
    r = run(
        [chunk("c1", "refund policy", "doc-1")],
        cited_doc_ids=["doc-1"],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].citation_status == CitationStatus.CITED


def test_chunk_citation_status_uncited():
    r = run(
        [chunk("c1", "refund policy", "doc-1"), chunk("c2", "other text", "doc-2")],
        cited_doc_ids=["doc-1"],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    statuses = {cp.chunk_id: cp.citation_status for cp in profile.chunks}
    assert statuses["c1"] == CitationStatus.CITED
    assert statuses["c2"] == CitationStatus.UNCITED


def test_chunk_citation_status_unknown_when_no_cited_ids():
    r = run([chunk("c1", "refund policy", "doc-1")])
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.chunks[0].citation_status == CitationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------

def test_overall_status_ok_when_no_issues():
    r = run(
        [chunk("c1", "refund policy covers all returns", "doc-1")],
        query="refund policy",
        cited_doc_ids=["doc-1"],
        corpus_entries=[corpus_entry("doc-1", age_days=10)],
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.overall_retrieval_status == "ok"


def test_overall_status_degraded_when_issues_present():
    r = run(
        [chunk("c1", "firmware kernel driver", "doc-1")],
        query="refund policy returns",
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.overall_retrieval_status == "degraded"


# ---------------------------------------------------------------------------
# run_id propagation
# ---------------------------------------------------------------------------

def test_run_id_propagated_to_profile():
    r = RAGRun(
        run_id="test-run-42",
        query="refund",
        retrieved_chunks=[chunk("c1", "refund details")],
        final_answer="Answer.",
    )
    profile = RetrievalEvidenceProfilerV0().build(r)
    assert profile.run_id == "test-run-42"
