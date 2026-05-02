"""
Integration tests: retrieval analyzers consuming RetrievalEvidenceProfile.

Covers:
- each analyzer works without profile (legacy path)
- each analyzer uses profile when present (profile path)
- evidence messages clearly state profile or fallback source
- analysis_source field is set correctly on every non-skip result
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    CitationStatus,
    FreshnessStatus,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk(chunk_id: str, text: str, source_doc_id: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id or chunk_id,
        score=None,
    )


def run_without_profile(
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
        retrieval_evidence_profile=None,
    )


def run_with_profile(
    chunks: list[RetrievedChunk],
    profile: RetrievalEvidenceProfile,
    *,
    query: str = "What is the refund policy?",
    cited_doc_ids: list[str] | None = None,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer="Answer.",
        cited_doc_ids=cited_doc_ids or [],
        retrieval_evidence_profile=profile,
    )


def chunk_profile(
    chunk_id: str,
    *,
    label: QueryRelevanceLabel = QueryRelevanceLabel.RELEVANT,
    freshness: FreshnessStatus = FreshnessStatus.VALID,
    citation: CitationStatus = CitationStatus.UNKNOWN,
    source_doc_id: str | None = None,
) -> ChunkEvidenceProfile:
    return ChunkEvidenceProfile(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id or chunk_id,
        query_relevance_label=label,
        relevance_method=RelevanceMethod.LEXICAL_OVERLAP,
        freshness_status=freshness,
        citation_status=citation,
    )


# ---------------------------------------------------------------------------
# RAGRun accepts retrieval_evidence_profile field
# ---------------------------------------------------------------------------

def test_ragrun_accepts_retrieval_evidence_profile():
    profile = RetrievalEvidenceProfile()
    r = RAGRun(
        query="test",
        retrieved_chunks=[chunk("c1", "text")],
        final_answer="Answer.",
        retrieval_evidence_profile=profile,
    )
    assert r.retrieval_evidence_profile is profile


def test_ragrun_retrieval_evidence_profile_defaults_to_none():
    r = RAGRun(
        query="test",
        retrieved_chunks=[chunk("c1", "text")],
        final_answer="Answer.",
    )
    assert r.retrieval_evidence_profile is None


# ---------------------------------------------------------------------------
# CitationMismatchAnalyzer
# ---------------------------------------------------------------------------

class TestCitationMismatchLegacy:
    def test_works_without_profile_pass(self):
        r = run_without_profile(
            [chunk("c1", "text", "doc-1")],
            cited_doc_ids=["doc-1"],
        )
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "legacy_heuristic_fallback"

    def test_works_without_profile_fail(self):
        r = run_without_profile(
            [chunk("c1", "text", "doc-1")],
            cited_doc_ids=["doc-1", "doc-ghost"],
        )
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.analysis_source == "legacy_heuristic_fallback"
        # legacy evidence strings are unchanged
        assert result.evidence == ["phantom citation: doc-ghost"]

    def test_skip_has_no_analysis_source(self):
        r = run_without_profile([])
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "skip"
        assert result.analysis_source is None


class TestCitationMismatchProfile:
    def test_uses_profile_when_present_pass(self):
        profile = RetrievalEvidenceProfile(phantom_citation_doc_ids=[])
        r = run_with_profile([chunk("c1", "text", "doc-1")], profile)
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_uses_profile_when_present_fail(self):
        profile = RetrievalEvidenceProfile(phantom_citation_doc_ids=["doc-ghost"])
        r = run_with_profile([chunk("c1", "text", "doc-1")], profile)
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.failure_type == FailureType.CITATION_MISMATCH
        assert result.stage == FailureStage.RETRIEVAL
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_evidence_has_profile_prefix(self):
        profile = RetrievalEvidenceProfile(phantom_citation_doc_ids=["doc-ghost"])
        r = run_with_profile([chunk("c1", "text", "doc-1")], profile)
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.evidence == ["[profile] phantom citation: doc-ghost"]

    def test_profile_ignores_run_cited_doc_ids(self):
        # Profile says no phantoms; run.cited_doc_ids has one outside retrieved set.
        # Profile path must win — result should be pass.
        profile = RetrievalEvidenceProfile(phantom_citation_doc_ids=[])
        r = run_with_profile(
            [chunk("c1", "text", "doc-1")],
            profile,
            cited_doc_ids=["doc-missing"],
        )
        result = CitationMismatchAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"


# ---------------------------------------------------------------------------
# ScopeViolationAnalyzer
# ---------------------------------------------------------------------------

class TestScopeViolationLegacy:
    def test_works_without_profile_pass(self):
        r = run_without_profile(
            [chunk("c1", "The refund policy covers returns.")],
            query="refund policy",
        )
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "legacy_heuristic_fallback"

    def test_works_without_profile_warn(self):
        r = run_without_profile(
            [
                chunk("c1", "Refund policy covers returns."),
                chunk("c2", "Kernel firmware drivers patch."),
            ],
            query="refund policy returns",
        )
        result = ScopeViolationAnalyzer({"min_overlap_ratio": 0.5}).analyze(r)
        assert result.status == "warn"
        assert result.analysis_source == "legacy_heuristic_fallback"
        # legacy evidence strings unchanged
        assert result.evidence == ["c2 overlap=0.00"]

    def test_skip_has_no_analysis_source(self):
        r = run_without_profile([])
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "skip"
        assert result.analysis_source is None


class TestScopeViolationProfile:
    def test_uses_profile_when_present_pass(self):
        profile = RetrievalEvidenceProfile(
            chunks=[chunk_profile("c1", label=QueryRelevanceLabel.RELEVANT)]
        )
        r = run_with_profile([chunk("c1", "refund policy")], profile)
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_warn_on_some_irrelevant(self):
        profile = RetrievalEvidenceProfile(
            chunks=[
                chunk_profile("c1", label=QueryRelevanceLabel.RELEVANT),
                chunk_profile("c2", label=QueryRelevanceLabel.IRRELEVANT),
            ]
        )
        r = run_with_profile(
            [chunk("c1", "refund policy"), chunk("c2", "firmware kernel")],
            profile,
        )
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "warn"
        assert result.failure_type == FailureType.SCOPE_VIOLATION
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_fail_when_all_irrelevant(self):
        profile = RetrievalEvidenceProfile(
            chunks=[
                chunk_profile("c1", label=QueryRelevanceLabel.IRRELEVANT),
                chunk_profile("c2", label=QueryRelevanceLabel.IRRELEVANT),
            ]
        )
        r = run_with_profile(
            [chunk("c1", "firmware"), chunk("c2", "kernel")],
            profile,
        )
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_evidence_has_profile_prefix(self):
        profile = RetrievalEvidenceProfile(
            chunks=[chunk_profile("c1", label=QueryRelevanceLabel.IRRELEVANT)]
        )
        r = run_with_profile([chunk("c1", "firmware kernel")], profile)
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.evidence == ["[profile] c1 label=irrelevant"]

    def test_profile_partial_label_does_not_trigger_warn(self):
        profile = RetrievalEvidenceProfile(
            chunks=[chunk_profile("c1", label=QueryRelevanceLabel.PARTIAL)]
        )
        r = run_with_profile([chunk("c1", "some text")], profile)
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "pass"

    def test_profile_skip_still_fires_on_empty_chunks(self):
        profile = RetrievalEvidenceProfile()
        r = run_with_profile([], profile)
        result = ScopeViolationAnalyzer().analyze(r)
        assert result.status == "skip"


# ---------------------------------------------------------------------------
# InconsistentChunksAnalyzer
# ---------------------------------------------------------------------------

class TestInconsistentChunksLegacy:
    def test_works_without_profile_pass(self):
        r = run_without_profile(
            [
                chunk("c1", "Refunds are available within 30 days."),
                chunk("c2", "Customers may request a refund within 30 days."),
            ]
        )
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "legacy_heuristic_fallback"

    def test_works_without_profile_warn(self):
        r = run_without_profile([
            chunk("c1", "The refund policy applies to hardware returns."),
            chunk("c2", "Refund policy does not apply to hardware returns."),
        ])
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "warn"
        assert result.analysis_source == "legacy_heuristic_fallback"
        # legacy evidence strings unchanged
        assert result.evidence == ["c1 <-> c2"]

    def test_skip_has_no_analysis_source(self):
        r = run_without_profile([])
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "skip"
        assert result.analysis_source is None


class TestInconsistentChunksProfile:
    def test_uses_profile_when_present_pass(self):
        profile = RetrievalEvidenceProfile(contradictory_pairs=[])
        r = run_with_profile(
            [chunk("c1", "Refunds available."), chunk("c2", "Returns accepted.")],
            profile,
        )
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_uses_profile_when_present_warn(self):
        profile = RetrievalEvidenceProfile(contradictory_pairs=[("c1", "c2")])
        r = run_with_profile(
            [chunk("c1", "Refunds available."), chunk("c2", "Returns accepted.")],
            profile,
        )
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "warn"
        assert result.failure_type == FailureType.INCONSISTENT_CHUNKS
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_evidence_has_profile_prefix(self):
        profile = RetrievalEvidenceProfile(contradictory_pairs=[("c1", "c2")])
        r = run_with_profile(
            [chunk("c1", "text a"), chunk("c2", "text b")],
            profile,
        )
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.evidence == ["[profile] c1 <-> c2"]

    def test_profile_uses_pairs_not_text_scan(self):
        # Profile says no contradictions even though text has negation signals.
        # Profile path must win — result should be pass.
        profile = RetrievalEvidenceProfile(contradictory_pairs=[])
        r = run_with_profile([
            chunk("c1", "The refund policy applies to hardware returns."),
            chunk("c2", "Refund policy does not apply to hardware returns."),
        ], profile)
        result = InconsistentChunksAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"


# ---------------------------------------------------------------------------
# StaleRetrievalAnalyzer
# ---------------------------------------------------------------------------

class TestStaleRetrievalLegacy:
    def test_works_without_profile_pass(self):
        fresh = CorpusEntry(
            doc_id="doc-1",
            text="text",
            timestamp=datetime.now(UTC) - timedelta(days=5),
        )
        r = run_without_profile(
            [chunk("c1", "text", "doc-1")],
            corpus_entries=[fresh],
        )
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "legacy_heuristic_fallback"

    def test_works_without_profile_fail(self):
        old = CorpusEntry(
            doc_id="doc-old",
            text="text",
            timestamp=datetime.now(UTC) - timedelta(days=200),
        )
        r = run_without_profile(
            [chunk("c1", "text", "doc-old")],
            corpus_entries=[old],
        )
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.analysis_source == "legacy_heuristic_fallback"
        # legacy evidence strings unchanged
        assert any("doc-old is" in e and "days old" in e for e in result.evidence)

    def test_skip_has_no_analysis_source(self):
        r = run_without_profile([])
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "skip"
        assert result.analysis_source is None


class TestStaleRetrievalProfile:
    def test_uses_profile_when_present_pass(self):
        profile = RetrievalEvidenceProfile(stale_doc_ids=[])
        r = run_with_profile([chunk("c1", "text", "doc-1")], profile)
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "pass"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_uses_profile_when_present_fail(self):
        profile = RetrievalEvidenceProfile(stale_doc_ids=["doc-old"])
        r = run_with_profile([chunk("c1", "text", "doc-old")], profile)
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.failure_type == FailureType.STALE_RETRIEVAL
        assert result.stage == FailureStage.RETRIEVAL
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_evidence_has_profile_prefix(self):
        profile = RetrievalEvidenceProfile(stale_doc_ids=["doc-old"])
        r = run_with_profile([chunk("c1", "text", "doc-old")], profile)
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.evidence == ["[profile] stale document: doc-old"]

    def test_profile_skips_corpus_entry_check(self):
        # Profile says doc-1 is stale; no corpus_entries provided.
        # Legacy would skip ("no corpus metadata available"), profile must win.
        profile = RetrievalEvidenceProfile(stale_doc_ids=["doc-1"])
        r = run_with_profile([chunk("c1", "text", "doc-1")], profile)
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "fail"
        assert result.analysis_source == "retrieval_evidence_profile"

    def test_profile_skip_still_fires_on_empty_chunks(self):
        profile = RetrievalEvidenceProfile(stale_doc_ids=["doc-old"])
        r = run_with_profile([], profile)
        result = StaleRetrievalAnalyzer().analyze(r)
        assert result.status == "skip"
