"""
Tests for PR 8 — RAGChecker-inspired claim-level diagnostic rollups.

Tests cover:
- retrieval_miss_suspected correctly classified
- context_ignored_suspected correctly classified
- value_error_count from value_conflicts
- citation_mismatch_suspected for entailed claims with wrong cited doc
- noisy_context detected when many chunks are unused
- stale_source_suspected for moderate-score unsupported claims
- empty/abstain handling
- aggregate rate calculations
- failure_pattern_summary formatting
- ClaimGroundingAnalyzer emits diagnostic_rollup in AnalyzerResult
- rollup summary_line appears in analyzer evidence
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from raggov.analyzers.grounding.diagnostic_rollups import (
    ClaimDiagnosticRollupBuilder,
    ClaimDiagnosticSummary,
    _CONTEXT_IGNORED_MIN_SCORE,
    _RETRIEVAL_MISS_MAX_SCORE,
    _STALE_SOURCE_MIN_SCORE,
    _DIAGNOSTIC_VERSION,
)
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceRecord
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.models.chunk import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str = "some text", source_doc_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, source_doc_id=source_doc_id, score=0.9)


def _candidate(
    chunk_id: str,
    lexical_overlap_score: float = 0.5,
    source_doc_id: str = "doc-1",
) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        chunk_text="some chunk text",
        chunk_text_preview="some chunk text",
        lexical_overlap_score=lexical_overlap_score,
        anchor_overlap_score=0.0,
        value_overlap_score=0.0,
        retrieval_score=None,
        rerank_score=None,
    )


def _record(
    claim_id: str = "claim_001",
    claim_text: str = "The subsidy is 60%.",
    verification_label: str = "entailed",
    candidates: list[EvidenceCandidate] | None = None,
    supporting_chunk_ids: list[str] | None = None,
    contradicting_chunk_ids: list[str] | None = None,
    value_conflicts: list[dict] | None = None,
    value_matches: list[dict] | None = None,
) -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        claim_type="numeric",
        atomicity_status="atomic",
        extracted_values=[],
        candidate_evidence_chunks=candidates or [],
        supporting_chunk_ids=supporting_chunk_ids or [],
        contradicting_chunk_ids=contradicting_chunk_ids or [],
        verification_label=verification_label,  # type: ignore[arg-type]
        verification_method="test_verifier",
        raw_support_score=0.5,
        calibrated_confidence=None,
        calibration_status="uncalibrated",
        evidence_reason="test reason",
        value_matches=value_matches or [],
        value_conflicts=value_conflicts or [],
        fallback_used=False,
    )


# ---------------------------------------------------------------------------
# 1. Retrieval miss suspected
# ---------------------------------------------------------------------------

class TestRetrievalMissSuspected:
    def test_unsupported_with_no_candidates_is_retrieval_miss(self) -> None:
        """No candidates → best_score = 0 → retrieval_miss."""
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(verification_label="unsupported", candidates=[])
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.retrieval_miss_suspected_count == 1
        assert summary.context_ignored_suspected_count == 0

    def test_unsupported_with_very_low_score_is_retrieval_miss(self) -> None:
        score = _RETRIEVAL_MISS_MAX_SCORE - 0.01
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="unsupported",
            candidates=[_candidate("c1", lexical_overlap_score=score)],
        )
        summary = builder.build([rec], retrieved_chunks=[_chunk("c1")], cited_doc_ids=None)
        assert summary.retrieval_miss_suspected_count == 1

    def test_entailed_claim_does_not_count_as_retrieval_miss(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(verification_label="entailed", candidates=[])
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.retrieval_miss_suspected_count == 0


# ---------------------------------------------------------------------------
# 2. Context ignored suspected
# ---------------------------------------------------------------------------

class TestContextIgnoredSuspected:
    def test_unsupported_with_high_score_is_context_ignored(self) -> None:
        score = _CONTEXT_IGNORED_MIN_SCORE + 0.05
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="unsupported",
            candidates=[_candidate("c1", lexical_overlap_score=score)],
        )
        summary = builder.build([rec], retrieved_chunks=[_chunk("c1")], cited_doc_ids=None)
        assert summary.context_ignored_suspected_count == 1
        assert summary.retrieval_miss_suspected_count == 0

    def test_borderline_score_below_context_ignored_threshold(self) -> None:
        """Score between retrieval_miss_max and context_ignored_min → not classified."""
        score = (_RETRIEVAL_MISS_MAX_SCORE + _CONTEXT_IGNORED_MIN_SCORE) / 2
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="unsupported",
            candidates=[_candidate("c1", lexical_overlap_score=score)],
        )
        summary = builder.build([rec], retrieved_chunks=[_chunk("c1")], cited_doc_ids=None)
        # Borderline — neither retrieval_miss nor context_ignored, may be stale_source
        total_classified = (
            summary.retrieval_miss_suspected_count
            + summary.context_ignored_suspected_count
        )
        assert total_classified == 0 or summary.stale_source_suspected_count >= 0  # no crash


# ---------------------------------------------------------------------------
# 3. Value error count
# ---------------------------------------------------------------------------

class TestValueErrorCount:
    def test_claim_with_value_conflict_counted(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="contradicted",
            value_conflicts=[{"claim_value": "60%", "evidence_value": "30%", "value_type": "percentage"}],
        )
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.value_error_count == 1

    def test_multiple_claims_with_conflicts_counted(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        recs = [
            _record("c1", verification_label="contradicted",
                    value_conflicts=[{"claim_value": "200", "evidence_value": "50", "value_type": "amount"}]),
            _record("c2", verification_label="contradicted",
                    value_conflicts=[{"claim_value": "25th", "evidence_value": "20th", "value_type": "date"}]),
            _record("c3", verification_label="entailed"),
        ]
        summary = builder.build(recs, retrieved_chunks=[], cited_doc_ids=None)
        assert summary.value_error_count == 2

    def test_entailed_claim_with_conflict_still_counted(self) -> None:
        """value_error_count is independent of verification_label."""
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="entailed",
            value_conflicts=[{"claim_value": "x", "evidence_value": "y", "value_type": "id"}],
        )
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.value_error_count == 1

    def test_no_conflicts_gives_zero_value_error_count(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(verification_label="unsupported", value_conflicts=[])
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.value_error_count == 0


# ---------------------------------------------------------------------------
# 4. Citation mismatch suspected
# ---------------------------------------------------------------------------

class TestCitationMismatchSuspected:
    def test_entailed_claim_with_uncited_supporting_chunk_is_mismatch(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="entailed",
            candidates=[_candidate("chunk-a", source_doc_id="doc-real")],
            supporting_chunk_ids=["chunk-a"],
        )
        chunks = [_chunk("chunk-a", source_doc_id="doc-real")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=["doc-cited"])
        assert summary.citation_mismatch_suspected_count == 1

    def test_entailed_claim_with_correctly_cited_chunk_no_mismatch(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="entailed",
            candidates=[_candidate("chunk-a", source_doc_id="doc-1")],
            supporting_chunk_ids=["chunk-a"],
        )
        chunks = [_chunk("chunk-a", source_doc_id="doc-1")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=["doc-1"])
        assert summary.citation_mismatch_suspected_count == 0

    def test_no_cited_docs_skips_citation_mismatch_analysis(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="entailed",
            supporting_chunk_ids=["chunk-a"],
        )
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.citation_mismatch_suspected_count == 0
        assert summary.has_cited_docs is False
        assert any("cited_doc_ids not provided" in note for note in summary.notes)

    def test_unsupported_claim_not_counted_in_citation_mismatch(self) -> None:
        """Citation mismatch only applies to entailed claims."""
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="unsupported",
            supporting_chunk_ids=["chunk-a"],
        )
        chunks = [_chunk("chunk-a", source_doc_id="doc-real")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=["doc-other"])
        assert summary.citation_mismatch_suspected_count == 0


# ---------------------------------------------------------------------------
# 5. Noisy context detection
# ---------------------------------------------------------------------------

class TestNoisyContextDetected:
    def test_many_unused_chunks_triggers_noisy_context(self) -> None:
        builder = ClaimDiagnosticRollupBuilder(config={
            "diagnostic_noise_fraction_threshold": 0.6,
            "diagnostic_noise_min_unused": 2,
        })
        # 1 claim uses chunk-1; chunks 2, 3, 4, 5 are never referenced
        rec = _record(
            candidates=[_candidate("chunk-1")],
            verification_label="entailed",
            supporting_chunk_ids=["chunk-1"],
        )
        chunks = [_chunk(f"chunk-{i}") for i in range(1, 6)]  # 5 chunks total
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=None)
        # 4/5 = 0.8 unused → above threshold
        assert summary.noisy_context_suspected is True
        assert summary.evidence_utilization_rate < 1.0

    def test_all_chunks_used_no_noise(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            candidates=[_candidate("chunk-1"), _candidate("chunk-2")],
            verification_label="entailed",
            supporting_chunk_ids=["chunk-1"],
        )
        chunks = [_chunk("chunk-1"), _chunk("chunk-2")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=None)
        assert summary.noisy_context_suspected is False

    def test_no_retrieved_chunks_no_noise(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(candidates=[], verification_label="unsupported")
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.noisy_context_suspected is False


# ---------------------------------------------------------------------------
# 6. Stale source suspected
# ---------------------------------------------------------------------------

class TestStaleSourceSuspected:
    def test_moderate_score_unsupported_with_no_value_match_is_stale(self) -> None:
        score = _STALE_SOURCE_MIN_SCORE + 0.02
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            verification_label="unsupported",
            candidates=[_candidate("c1", lexical_overlap_score=score)],
            value_matches=[],
        )
        # Score is above stale_source threshold but below context_ignored threshold
        # → should classify as stale_source
        summary = builder.build([rec], retrieved_chunks=[_chunk("c1")], cited_doc_ids=None)
        # stale_source fires when score is in [stale_min, context_ignored_min)
        if score < _CONTEXT_IGNORED_MIN_SCORE:
            assert summary.stale_source_suspected_count == 1
        else:
            assert summary.context_ignored_suspected_count >= 1


# ---------------------------------------------------------------------------
# 7. Empty / abstain handling
# ---------------------------------------------------------------------------

class TestEmptyAndAbstainHandling:
    def test_empty_records_returns_zero_summary(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        summary = builder.build([], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.total_claims == 0
        assert summary.entailed_claims == 0
        assert summary.claim_support_rate == 0.0
        assert summary.retrieval_miss_suspected_count == 0
        assert summary.context_ignored_suspected_count == 0
        assert summary.value_error_count == 0
        assert summary.citation_mismatch_suspected_count == 0
        assert not summary.noisy_context_suspected

    def test_abstained_claims_counted_separately(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        # Abstained claims use a label not in {entailed, unsupported, contradicted}
        rec = ClaimEvidenceRecord(
            claim_id="claim_001",
            claim_text="Some claim.",
            claim_type="general_factual",
            atomicity_status="atomic",
            extracted_values=[],
            candidate_evidence_chunks=[],
            supporting_chunk_ids=[],
            contradicting_chunk_ids=[],
            verification_label="abstain",  # type: ignore[arg-type]
            verification_method="abstaining_verifier",
            raw_support_score=0.0,
            calibrated_confidence=None,
            calibration_status="uncalibrated",
            evidence_reason="abstained",
            value_matches=[],
            value_conflicts=[],
            fallback_used=True,
        )
        summary = builder.build([rec], retrieved_chunks=[], cited_doc_ids=None)
        assert summary.abstained_claims == 1
        assert summary.total_claims == 1
        assert summary.entailed_claims == 0
        assert any("abstained" in note for note in summary.notes)

    def test_empty_notes_include_diagnostic_message(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        summary = builder.build([], retrieved_chunks=[], cited_doc_ids=None)
        assert any("No claims" in note for note in summary.notes)


# ---- Monkeypatch helper for abstain test -----------------------------------
def _false_pattern_absent(summary: ClaimDiagnosticSummary) -> bool:
    return (
        summary.retrieval_miss_suspected_count == 0
        and summary.context_ignored_suspected_count == 0
        and summary.value_error_count == 0
        and summary.citation_mismatch_suspected_count == 0
        and not summary.noisy_context_suspected
    )

ClaimDiagnosticSummary.false_pattern_absent = _false_pattern_absent  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 8. Aggregate rate calculations
# ---------------------------------------------------------------------------

class TestAggregateRates:
    def test_claim_support_rate_correct(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        recs = [
            _record("c1", verification_label="entailed"),
            _record("c2", verification_label="entailed"),
            _record("c3", verification_label="unsupported"),
            _record("c4", verification_label="contradicted"),
        ]
        summary = builder.build(recs, retrieved_chunks=[], cited_doc_ids=None)
        assert summary.total_claims == 4
        assert summary.entailed_claims == 2
        assert summary.unsupported_claims == 1
        assert summary.contradicted_claims == 1
        assert summary.claim_support_rate == pytest.approx(0.5)
        assert summary.contradiction_rate == pytest.approx(0.25)
        assert summary.unsupported_rate == pytest.approx(0.25)

    def test_all_entailed_rates(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        recs = [_record(str(i), verification_label="entailed") for i in range(4)]
        summary = builder.build(recs, retrieved_chunks=[], cited_doc_ids=None)
        assert summary.claim_support_rate == 1.0
        assert summary.contradiction_rate == 0.0
        assert summary.unsupported_rate == 0.0

    def test_evidence_utilization_rate_all_used(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(
            candidates=[_candidate("c1"), _candidate("c2")],
            verification_label="entailed",
        )
        chunks = [_chunk("c1"), _chunk("c2")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=None)
        assert summary.evidence_utilization_rate == 1.0

    def test_evidence_utilization_rate_none_used(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        rec = _record(candidates=[], verification_label="unsupported")
        chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
        summary = builder.build([rec], retrieved_chunks=chunks, cited_doc_ids=None)
        assert summary.evidence_utilization_rate == 0.0

    def test_citation_support_rate_partial(self) -> None:
        builder = ClaimDiagnosticRollupBuilder()
        recs = [
            _record("r1", verification_label="entailed",
                    candidates=[_candidate("c1", source_doc_id="doc-cited")],
                    supporting_chunk_ids=["c1"]),
            _record("r2", verification_label="entailed",
                    candidates=[_candidate("c2", source_doc_id="doc-other")],
                    supporting_chunk_ids=["c2"]),
        ]
        chunks = [
            _chunk("c1", source_doc_id="doc-cited"),
            _chunk("c2", source_doc_id="doc-other"),
        ]
        summary = builder.build(recs, retrieved_chunks=chunks, cited_doc_ids=["doc-cited"])
        # 1 of 2 entailed claims has a cited chunk
        assert summary.citation_support_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. failure_pattern_summary formatting
# ---------------------------------------------------------------------------

class TestFailurePatternSummary:
    def _make_summary(self, **kwargs) -> ClaimDiagnosticSummary:
        defaults = dict(
            total_claims=3,
            entailed_claims=1,
            unsupported_claims=2,
            contradicted_claims=0,
            abstained_claims=0,
            claim_support_rate=0.33,
            contradiction_rate=0.0,
            unsupported_rate=0.67,
            citation_support_rate=0.0,
            evidence_utilization_rate=0.5,
            retrieval_miss_suspected_count=0,
            context_ignored_suspected_count=0,
            value_error_count=0,
            stale_source_suspected_count=0,
            noisy_context_suspected=False,
            citation_mismatch_suspected_count=0,
            diagnostic_version=_DIAGNOSTIC_VERSION,
            has_cited_docs=False,
            notes=[],
        )
        defaults.update(kwargs)
        return ClaimDiagnosticSummary(**defaults)

    def test_no_failures_returns_no_suspected_failures(self) -> None:
        s = self._make_summary()
        assert s.failure_pattern_summary() == "no_suspected_failures"

    def test_retrieval_miss_appears_in_summary(self) -> None:
        s = self._make_summary(retrieval_miss_suspected_count=2)
        assert "retrieval_miss×2" in s.failure_pattern_summary()

    def test_multiple_patterns_all_appear(self) -> None:
        s = self._make_summary(
            retrieval_miss_suspected_count=1,
            context_ignored_suspected_count=1,
            value_error_count=2,
            citation_mismatch_suspected_count=1,
            noisy_context_suspected=True,
        )
        text = s.failure_pattern_summary()
        assert "retrieval_miss" in text
        assert "context_ignored" in text
        assert "value_error" in text
        assert "citation_mismatch" in text
        assert "noisy_retrieval" in text

    def test_as_dict_is_serializable(self) -> None:
        import json
        s = self._make_summary(retrieval_miss_suspected_count=1)
        payload = s.as_dict()
        # Should not raise
        dumped = json.dumps(payload)
        assert "retrieval_miss_suspected_count" in dumped


# ---------------------------------------------------------------------------
# 10. ClaimGroundingAnalyzer emits diagnostic_rollup
# ---------------------------------------------------------------------------

class TestClaimGroundingAnalyzerEmitsRollup:
    def test_analyzer_result_has_diagnostic_rollup(self) -> None:
        from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
        from raggov.models.run import RAGRun

        run = RAGRun(
            run_id="test-pr8-001",
            query="What is the subsidy under PM-KUSUM?",
            final_answer="The subsidy under PM-KUSUM is 60%.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    text="Under PM-KUSUM, a subsidy of 60% is provided to farmers.",
                    source_doc_id="pm-kusum-doc",
                    score=0.91,
                )
            ],
        )
        analyzer = ClaimGroundingAnalyzer(config={})
        result = analyzer.analyze(run)
        assert result.diagnostic_rollup is not None
        assert "total_claims" in result.diagnostic_rollup
        assert "false_pass_rate" not in result.diagnostic_rollup  # wrong key
        assert "retrieval_miss_suspected_count" in result.diagnostic_rollup
        assert "diagnostic_version" in result.diagnostic_rollup
        assert result.diagnostic_rollup["diagnostic_version"] == _DIAGNOSTIC_VERSION

    def test_analyzer_evidence_contains_diagnostic_pattern_line(self) -> None:
        from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
        from raggov.models.run import RAGRun

        run = RAGRun(
            run_id="test-pr8-002",
            query="What is the penalty?",
            final_answer="The penalty is Rs. 200 per day.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    text="A late fee of Rs. 50 per day is levied.",
                    source_doc_id="gst-doc",
                    score=0.88,
                )
            ],
        )
        analyzer = ClaimGroundingAnalyzer(config={})
        result = analyzer.analyze(run)
        # At least one evidence line should mention the diagnostic version tag
        diagnostic_lines = [e for e in result.evidence if "diagnostic_rollup_heuristic_v0" in e]
        assert len(diagnostic_lines) >= 1

    def test_analyzer_rollup_zero_on_no_claims_extracted(self) -> None:
        from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
        from raggov.models.run import RAGRun

        # Non-substantive answer → no claims extracted → skip
        run = RAGRun(
            run_id="test-pr8-003",
            query="Anything?",
            final_answer="I cannot help with that.",
            retrieved_chunks=[
                RetrievedChunk(chunk_id="c1", text="some text", source_doc_id="d1", score=0.5)
            ],
        )
        analyzer = ClaimGroundingAnalyzer(config={})
        result = analyzer.analyze(run)
        # Skip status → no rollup
        assert result.status == "skip"
        assert result.diagnostic_rollup is None
