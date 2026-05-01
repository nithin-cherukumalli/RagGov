"""Tests for citation faithfulness probing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from raggov.analyzers.grounding.citation_faithfulness import CitationFaithfulnessProbe
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceRecord


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def chunk(
    chunk_id: str,
    text: str,
    *,
    source_doc_id: str | None = None,
    score: float | None = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id or f"doc-{chunk_id}",
        score=score,
    )


def run_with_answer(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is the policy?",
    cited_doc_ids: list[str] | None = None,
    answer_confidence: float | None = 0.8,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer=answer,
        cited_doc_ids=cited_doc_ids or [],
        answer_confidence=answer_confidence,
    )


def build_record(
    claim_id: str,
    claim_text: str,
    verification_label: str,
    supporting_chunk_ids: list[str],
    evidence_reason: str = "Matched",
    fallback_used: bool = False,
) -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        claim_type="general_factual",
        atomicity_status="atomic",
        extracted_values=[],
        candidate_evidence_chunks=[],
        supporting_chunk_ids=supporting_chunk_ids,
        contradicting_chunk_ids=[],
        verification_label=verification_label,
        verification_method="heuristic",
        raw_support_score=1.0,
        calibrated_confidence=None,
        calibration_status="uncalibrated",
        evidence_reason=evidence_reason,
        value_matches=[],
        value_conflicts=[],
        fallback_used=fallback_used,
    )


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_correct_claim_with_correct_citation_passes(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["The policy requires approval."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="The policy requires approval.",
        verification_label="entailed",
        supporting_chunk_ids=["chunk-1"],
    )

    run = run_with_answer(
        "The policy requires approval.",
        [chunk("chunk-1", "The policy requires approval.", source_doc_id="doc-1")],
        cited_doc_ids=["doc-1"],
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None
    assert result.citation_probe_results is not None
    assert result.citation_probe_results[0]["status"] == "citation_supported"


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_correct_claim_with_wrong_citation_fails(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["The policy allows refunds within 30 days."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="The policy allows refunds within 30 days.",
        verification_label="entailed",
        supporting_chunk_ids=["chunk-1"],  # Maps to doc-1
        evidence_reason="Matched doc-1",
    )

    run = run_with_answer(
        "The policy allows refunds within 30 days.",
        [
            chunk("chunk-1", "Refunds allowed within 30 days.", source_doc_id="doc-1"),
            chunk("chunk-2", "Refunds in 45 days.", source_doc_id="doc-2"),
        ],
        cited_doc_ids=["doc-2"],  # Cited the wrong document
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION
    assert result.citation_probe_results[0]["status"] == "citation_mismatch"


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_unsupported_claim_with_citation_fails(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["Alpha protocol requires a xenolith seal."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="Alpha protocol requires a xenolith seal.",
        verification_label="unsupported",
        supporting_chunk_ids=[],
        evidence_reason="No match",
    )

    run = run_with_answer(
        "Alpha protocol requires a xenolith seal.",
        [chunk("chunk-1", "Text", source_doc_id="doc-1")],
        cited_doc_ids=["doc-1"],
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION
    assert result.citation_probe_results[0]["status"] == "unsupported_cited_claim"


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_supported_claim_with_no_citation_warns(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["The policy requires approval."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="The policy requires approval.",
        verification_label="entailed",
        supporting_chunk_ids=["chunk-1"],
    )

    run = run_with_answer(
        "The policy requires approval.",
        [chunk("chunk-1", "The policy requires approval.", source_doc_id="doc-1")],
        cited_doc_ids=[],  # No citations provided
    )

    # Note: Because warn_claims > 0 (1 missing citation), the result status should be "warn"
    # if it's under suspicious_threshold, or "fail" if over. The default suspicious_threshold is 2.
    result = CitationFaithfulnessProbe({"suspicious_threshold": 2}).analyze(run)

    assert result.status == "warn"
    assert result.citation_probe_results[0]["status"] == "citation_missing"


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_no_citations_and_low_support_does_not_falsely_call_post_rationalization(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["This is a guess."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="This is a guess.",
        verification_label="unsupported",
        supporting_chunk_ids=[],
        evidence_reason="No match",
    )

    run = run_with_answer(
        "This is a guess.",
        [chunk("chunk-1", "Text", source_doc_id="doc-1")],
        cited_doc_ids=[],
        answer_confidence=0.4, # Not confident
    )

    result = CitationFaithfulnessProbe().analyze(run)

    # With no citations and confidence below threshold, the probe correctly skips.
    # The grounding analyzer (ClaimGroundingAnalyzer) handles the unsupported claim detection.
    assert result.status == "skip"


@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimExtractor.extract")
@patch("raggov.analyzers.grounding.citation_faithfulness.ClaimEvidenceBuilder._build_single")
def test_fallback_visibility_preserved(mock_build: MagicMock, mock_extract: MagicMock) -> None:
    mock_extract.return_value = ["The policy requires approval."]
    mock_build.return_value = build_record(
        claim_id="claim-1",
        claim_text="The policy requires approval.",
        verification_label="entailed",
        supporting_chunk_ids=["chunk-1"],
        evidence_reason="Matched via fallback",
        fallback_used=True,
    )

    run = run_with_answer(
        "The policy requires approval.",
        [chunk("chunk-1", "The policy requires approval.", source_doc_id="doc-1")],
        cited_doc_ids=["doc-1"],
    )

    result = CitationFaithfulnessProbe().analyze(run)
    assert result.citation_probe_results[0]["fallback_used"] is True


def test_probe_skips_without_answer_or_chunks() -> None:
    no_answer = CitationFaithfulnessProbe().analyze(
        run_with_answer("", [chunk("chunk-1", "Text", source_doc_id="doc-1")], cited_doc_ids=["doc-1"])
    )
    no_chunks = CitationFaithfulnessProbe().analyze(
        run_with_answer("A complete answer exists.", [], cited_doc_ids=["doc-1"])
    )

    assert no_answer.status == "skip"
    assert no_answer.evidence == ["no final answer to probe"]
    assert no_chunks.status == "skip"
    assert no_chunks.evidence == ["no retrieved chunks available"]

