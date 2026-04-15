"""Tests for the sufficiency analyzer."""

from __future__ import annotations

from typing import Any

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


class ChatClient:
    """Fake LLM client exposing a chat method."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class CompleteClient:
    """Fake LLM client exposing a complete method."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {"text": self.response}


class FailingClient:
    """Fake LLM client that raises."""

    def chat(self, prompt: str) -> str:
        raise RuntimeError("client unavailable")


def chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text=text,
        source_doc_id="doc-1",
        score=None,
    )


def run_with_context(query: str, chunks: list[RetrievedChunk]) -> RAGRun:
    return RAGRun(query=query, retrieved_chunks=chunks, final_answer="Answer.")


def test_deterministic_mode_fails_when_context_has_low_term_coverage() -> None:
    run = run_with_context(
        "refund policy warranty",
        [chunk("Refund details are available for returns.")],
    )

    result = SufficiencyAnalyzer({"min_coverage_ratio": 0.8}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert result.evidence == [
        "Query term coverage: 33%. Terms not found in context: policy, warranty"
    ]
    assert result.remediation == (
        "Context does not cover key query terms. Consider expanding retrieval "
        "(increase top-k), broadening the query, or abstaining. should_abstain=True"
    )


def test_deterministic_mode_passes_when_context_covers_query_terms() -> None:
    run = run_with_context(
        "refund policy",
        [chunk("The refund policy explains returns and credits.")],
    )

    result = SufficiencyAnalyzer({"min_coverage_ratio": 1.0}).analyze(run)

    assert result.status == "pass"
    assert result.evidence == ["Query term coverage: 100%. Terms not found in context: none"]


def test_sufficiency_skips_empty_chunks_or_empty_query_terms() -> None:
    analyzer = SufficiencyAnalyzer()

    no_chunks = analyzer.analyze(run_with_context("refund policy", []))
    no_terms = analyzer.analyze(run_with_context("and or the", [chunk("context")]))

    assert no_chunks.status == "skip"
    assert no_chunks.evidence == ["no retrieved chunks available"]
    assert no_terms.status == "skip"
    assert no_terms.evidence == ["no meaningful query terms available"]


def test_llm_mode_fails_when_judge_says_context_is_insufficient() -> None:
    client = ChatClient(
        '{"sufficient": false, "missing": "warranty terms", "confidence": 0.9}'
    )
    run = run_with_context("refund warranty", [chunk("Refund policy only.")])

    result = SufficiencyAnalyzer({"use_llm": True, "llm_client": client}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert result.evidence == ["LLM judge missing: warranty terms"]
    assert result.remediation == (
        "Context does not cover key query terms. Consider expanding retrieval "
        "(increase top-k), broadening the query, or abstaining. should_abstain=True"
    )
    assert "Given this query: refund warranty" in client.prompts[0]
    assert "Answer with JSON" in client.prompts[0]


def test_llm_mode_warns_when_sufficient_but_confidence_is_low() -> None:
    client = CompleteClient(
        '{"sufficient": true, "missing": "", "confidence": 0.5}'
    )
    run = run_with_context("refund policy", [chunk("Refund policy details.")])

    result = SufficiencyAnalyzer({"use_llm": True, "llm_client": client}).analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.SUFFICIENCY
    assert result.evidence == ["LLM judge confidence: 0.50"]


def test_llm_mode_passes_when_sufficient_with_adequate_confidence() -> None:
    client = ChatClient(
        '{"sufficient": true, "missing": "", "confidence": 0.8}'
    )
    run = run_with_context("refund policy", [chunk("Refund policy details.")])

    result = SufficiencyAnalyzer({"use_llm": True, "llm_client": client}).analyze(run)

    assert result.status == "pass"
    assert result.evidence == ["LLM judge confidence: 0.80"]


def test_llm_mode_falls_back_to_deterministic_when_client_fails() -> None:
    run = run_with_context("refund warranty", [chunk("Refund policy only.")])

    result = SufficiencyAnalyzer(
        {
            "use_llm": True,
            "llm_client": FailingClient(),
            "min_coverage_ratio": 0.8,
        }
    ).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.evidence == [
        "LLM sufficiency judge failed; fell back to deterministic mode: client unavailable",
        "Query term coverage: 50%. Terms not found in context: warranty",
    ]
