"""Tests for v1 structured SufficiencyResult fields."""

from __future__ import annotations

from typing import Any

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    EvidenceCoverage,
    EvidenceRequirement,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


class FailingClient:
    def chat(self, prompt: str) -> str:
        raise RuntimeError("judge unavailable")


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
    )


def run_with_context(query: str, chunks: list[RetrievedChunk]) -> RAGRun:
    return RAGRun(query=query, retrieved_chunks=chunks, final_answer="Answer.")


def test_no_chunks_returns_unknown() -> None:
    run = run_with_context("AP school optional holidays", [])

    result = SufficiencyAnalyzer().analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.sufficiency_label == "unknown"
    assert result.sufficiency_result.should_abstain is False
    assert any("no retrieved chunks" in item for item in result.sufficiency_result.limitations)
    assert "heuristic_v0" in result.sufficiency_result.method


def test_low_coverage_returns_insufficient() -> None:
    run = run_with_context(
        "AP school optional holidays 2025 academic calendar",
        [chunk("chunk-1", "teacher transfers seniority roster")],
    )

    result = SufficiencyAnalyzer().analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.sufficiency_label == "insufficient"
    assert result.sufficiency_result.should_abstain is True
    assert result.sufficiency_result.should_expand_retrieval is True
    assert result.sufficiency_result.threshold_used == 0.3
    assert result.sufficiency_result.coverage


def test_high_coverage_without_claims_returns_unknown() -> None:
    run = run_with_context(
        "GAD optional holidays 2025",
        [
            chunk(
                "chunk-1",
                "GAD optional holidays circular 2025 general administration",
            )
        ],
    )

    result = SufficiencyAnalyzer().analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.sufficiency_label == "unknown"
    assert result.sufficiency_result.should_abstain is False
    assert (
        "Cannot distinguish sufficient context from term overlap"
        in result.sufficiency_result.limitations
    )


def test_claim_sidecar_unsupported_returns_partial() -> None:
    run = run_with_context(
        "GAD optional holidays 2025",
        [chunk("chunk-1", "GAD optional holidays circular 2025")],
    )
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        claim_results=[
            ClaimResult(
                claim_text="Optional holiday applies to all AP schools.",
                label="unsupported",
                supporting_chunk_ids=[],
            )
        ],
    )

    result = SufficiencyAnalyzer({"prior_results": [prior]}).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.sufficiency_label == "partial"
    assert "Optional holiday applies to all AP schools." in result.sufficiency_result.missing_evidence
    assert "Optional holiday applies to all AP schools." in result.sufficiency_result.affected_claims
    assert "claim_grounding_sidecar" in result.sufficiency_result.method


def test_claim_sidecar_contradicted_populates_affected() -> None:
    run = run_with_context(
        "GAD optional holidays 2025",
        [chunk("chunk-1", "GAD optional holidays circular 2025")],
    )
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        claim_results=[
            ClaimResult(
                claim_text="The circular applies to private banks.",
                label="contradicted",
                contradicting_chunk_ids=["chunk-1"],
            )
        ],
    )

    result = SufficiencyAnalyzer({"prior_results": [prior]}).analyze(run)

    assert result.sufficiency_result is not None
    assert "The circular applies to private banks." in result.sufficiency_result.affected_claims
    assert any(
        item.status == "contradicted"
        for item in result.sufficiency_result.coverage
    )


def test_fallback_logged_in_limitations() -> None:
    run = run_with_context(
        "AP school optional holidays 2025 academic calendar",
        [chunk("chunk-1", "teacher transfers seniority roster")],
    )

    result = SufficiencyAnalyzer(
        {"use_llm": True, "llm_client": FailingClient()}
    ).analyze(run)

    assert result.sufficiency_result is not None
    assert result.sufficiency_result.fallback_used is True
    assert any("LLM judge failed" in item for item in result.sufficiency_result.limitations)
    assert "heuristic_v0" in result.sufficiency_result.method


def test_schema_round_trips_json() -> None:
    result = SufficiencyResult(
        sufficient=False,
        sufficiency_label="partial",
        required_evidence=[
            EvidenceRequirement(
                requirement_id="req-1",
                description="Need the operative date.",
                requirement_type="date",
                importance="critical",
                query_span="2025",
            )
        ],
        coverage=[
            EvidenceCoverage(
                requirement_id="req-1",
                status="partial",
                supporting_chunk_ids=["chunk-1"],
                contradicting_chunk_ids=["chunk-2"],
                rationale="Date appears but exception is missing.",
                verifier="heuristic",
                confidence=0.5,
            )
        ],
        should_expand_retrieval=True,
        should_abstain=False,
        threshold_used=0.3,
        fallback_used=True,
        limitations=["limit"],
        missing_evidence=["missing"],
        affected_claims=["claim"],
        evidence_chunk_ids=["chunk-1"],
        method="term_coverage_heuristic_v0",
        calibration_status="uncalibrated",
    )

    restored = SufficiencyResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_backward_compatible_fields_still_present() -> None:
    run = run_with_context(
        "AP school optional holidays 2025 academic calendar",
        [chunk("chunk-1", "teacher transfers seniority roster")],
    )

    result = SufficiencyAnalyzer().analyze(run)
    payload: dict[str, Any] = result.sufficiency_result.model_dump()  # type: ignore[union-attr]

    for field in (
        "sufficient",
        "missing_evidence",
        "affected_claims",
        "evidence_chunk_ids",
        "method",
        "calibration_status",
    ):
        assert field in payload
