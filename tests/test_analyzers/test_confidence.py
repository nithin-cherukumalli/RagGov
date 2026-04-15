"""Tests for confidence signal aggregation."""

from __future__ import annotations

from raggov.analyzers.confidence.confidence import ConfidenceAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


def chunk(chunk_id: str, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="retrieved context",
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


def run_with_confidence(
    answer_confidence: float | None = None,
    chunks: list[RetrievedChunk] | None = None,
) -> RAGRun:
    return RAGRun(
        query="query",
        retrieved_chunks=chunks or [],
        final_answer="answer",
        answer_confidence=answer_confidence,
    )


def test_confidence_passes_with_no_penalties() -> None:
    result = ConfidenceAnalyzer().analyze(
        run_with_confidence(chunks=[chunk("chunk-1", 0.9), chunk("chunk-2", 0.8)])
    )

    assert result.status == "pass"
    assert result.score == 1.0
    assert result.evidence == [
        "base score: 1.00",
        "average retrieval score 0.85: no penalty",
        "final score: 1.00",
    ]


def test_confidence_warns_for_mid_score_after_prior_penalties() -> None:
    prior_results = [
        AnalyzerResult(analyzer_name="scope", status="warn"),
        AnalyzerResult(analyzer_name="citation", status="skip"),
    ]
    result = ConfidenceAnalyzer(
        {
            "prior_results": prior_results,
            "warn_confidence_threshold": 0.9,
        }
    ).analyze(run_with_confidence(answer_confidence=0.7))

    assert result.status == "warn"
    assert result.failure_type == FailureType.LOW_CONFIDENCE
    assert result.stage == FailureStage.CONFIDENCE
    assert result.score == 0.7
    assert result.evidence == [
        "base score: 1.00",
        "blended caller answer_confidence 0.70: score 0.85",
        "prior result scope status warn: -0.10",
        "prior result citation status skip: -0.05",
        "final score: 0.70",
    ]


def test_confidence_fails_when_score_below_low_threshold() -> None:
    prior_results = [
        AnalyzerResult(analyzer_name="grounding", status="fail"),
        AnalyzerResult(analyzer_name="security", status="fail"),
    ]
    result = ConfidenceAnalyzer({"prior_results": prior_results}).analyze(
        run_with_confidence(answer_confidence=0.2, chunks=[chunk("chunk-1", 0.3)])
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.LOW_CONFIDENCE
    assert result.stage == FailureStage.CONFIDENCE
    assert result.score == 0.1
    assert result.evidence == [
        "base score: 1.00",
        "blended caller answer_confidence 0.20: score 0.60",
        "prior result grounding status fail: -0.20",
        "prior result security status fail: -0.20",
        "average retrieval score 0.30: -0.10",
        "final score: 0.10",
    ]
    assert result.remediation == (
        "Confidence too low to trust output. Consider abstaining, re-retrieving, "
        "or requesting human review."
    )


def test_confidence_score_is_clamped_to_zero() -> None:
    prior_results = [
        AnalyzerResult(analyzer_name=f"fail-{index}", status="fail")
        for index in range(10)
    ]

    result = ConfidenceAnalyzer({"prior_results": prior_results}).analyze(
        run_with_confidence(answer_confidence=0.0, chunks=[chunk("chunk-1", 0.1)])
    )

    assert result.status == "fail"
    assert result.score == 0.0
    assert result.evidence[-1] == "final score: 0.00"
