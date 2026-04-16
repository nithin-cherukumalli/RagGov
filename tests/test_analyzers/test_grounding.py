"""Tests for grounding claim extraction and support analysis."""

from __future__ import annotations

import json
from typing import Any

from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


class ClaimClient:
    """Fake LLM client for claim extraction."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class GroundingClient:
    """Fake LLM client for claim grounding."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FailingClaimClient:
    """Fake LLM client that raises."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("extractor unavailable")


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
    )


def run_with_answer(answer: str, chunks: list[RetrievedChunk]) -> RAGRun:
    return RAGRun(query="What is covered?", retrieved_chunks=chunks, final_answer=answer)


def test_claim_extractor_deterministic_splits_sentences_and_filters_short_claims() -> None:
    answer = (
        "Hi there. The refund policy covers hardware returns for thirty days! "
        "Warranty support requires an active service plan? Too short."
    )

    claims = ClaimExtractor().extract(answer)

    assert claims == [
        "The refund policy covers hardware returns for thirty days!",
        "Warranty support requires an active service plan?",
    ]


def test_claim_extractor_llm_mode_parses_json_array() -> None:
    client = ClaimClient('["Refunds last thirty days.", "Warranty requires service."]')
    extractor = ClaimExtractor(use_llm=True, llm_client=client)

    claims = extractor.extract("Refunds last thirty days. Warranty requires service.")

    assert claims == ["Refunds last thirty days.", "Warranty requires service."]
    assert "Return only a JSON array of strings" in client.prompts[0]


def test_claim_extractor_llm_mode_falls_back_to_deterministic_on_failure() -> None:
    extractor = ClaimExtractor(use_llm=True, llm_client=FailingClaimClient())

    claims = extractor.extract("The refund policy covers hardware returns for thirty days.")

    assert claims == ["The refund policy covers hardware returns for thirty days."]


def test_claim_grounding_passes_when_all_claims_are_entailed() -> None:
    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.status == "pass"
    assert result.evidence == [
        '{"claim_text":"The refund policy covers hardware returns for thirty days.",'
        '"label":"entailed","supporting_chunk_ids":["chunk-1"],"confidence":1.0}'
    ]


def test_claim_grounding_warns_for_unsupported_claim_below_fail_threshold() -> None:
    run = run_with_answer(
        (
            "The refund policy covers hardware returns for thirty days. "
            "Warranty support includes international on-site repairs."
        ),
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer({"fail_threshold": 0.75}).analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GROUNDING
    assert result.remediation == (
        "1 of 2 claims are unsupported by retrieved context. "
        "Review retrieval quality or add source verification."
    )


def test_claim_grounding_fails_when_unsupported_fraction_reaches_threshold() -> None:
    run = run_with_answer(
        (
            "The refund policy covers hardware returns for thirty days. "
            "Warranty support includes international on-site repairs."
        ),
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer({"fail_threshold": 0.5}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GROUNDING


def test_claim_grounding_contradiction_is_always_at_least_warn() -> None:
    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy does not cover hardware returns.")],
    )

    result = ClaimGroundingAnalyzer({"fail_threshold": 1.1}).analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.CONTRADICTED_CLAIM
    assert result.stage == FailureStage.GROUNDING


def test_claim_grounding_skips_without_chunks_or_claims() -> None:
    analyzer = ClaimGroundingAnalyzer()

    no_chunks = analyzer.analyze(run_with_answer("A detailed answer exists here.", []))
    no_claims = analyzer.analyze(run_with_answer("Too short.", [chunk("chunk-1", "text")]))

    assert no_chunks.status == "skip"
    assert no_chunks.evidence == ["no retrieved chunks available"]
    assert no_claims.status == "skip"
    assert no_claims.evidence == ["no claims extracted from final answer"]


def test_claim_grounding_llm_mode_parses_claim_results() -> None:
    client = GroundingClient(
        [
            (
                '{"label":"entailed","confidence":0.9,'
                '"evidence_chunk_id":"chunk-1"}'
            )
        ]
    )
    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer({"use_llm": True, "llm_client": client}).analyze(run)

    assert result.status == "pass"
    assert result.evidence == [
        '{"claim_text":"The refund policy covers hardware returns for thirty days.",'
        '"label":"entailed","supporting_chunk_ids":["chunk-1"],"confidence":0.9}'
    ]
    assert "support, contradict, or neither support nor contradict" in client.prompts[0]


def test_claim_grounding_llm_mode_falls_back_per_claim_on_failure() -> None:
    class FailingGroundingClient:
        def chat(self, prompt: str) -> dict[str, Any]:
            raise RuntimeError("judge unavailable")

    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer(
        {"use_llm": True, "llm_client": FailingGroundingClient()}
    ).analyze(run)

    assert result.status == "pass"
    assert "entailed" in result.evidence[0]


def test_claim_grounding_deterministic_supports_paraphrased_numeric_claim() -> None:
    analyzer = ClaimGroundingAnalyzer()

    result = analyzer._evaluate_claim_deterministic(
        "Revenue grew 15% YoY.",
        [chunk("chunk-1", "Revenue increased 15% annually.")],
    )

    assert result.label == "entailed"
    assert result.supporting_chunk_ids == ["chunk-1"]
    assert result.confidence is not None
    assert result.confidence >= 0.5


def test_claim_grounding_deterministic_requires_numeric_anchor_match() -> None:
    analyzer = ClaimGroundingAnalyzer()

    result = analyzer._evaluate_claim_deterministic(
        "Revenue grew 15% YoY.",
        [chunk("chunk-1", "Revenue grew 12% annually.")],
    )

    assert result.label == "unsupported"
    assert result.supporting_chunk_ids == ["chunk-1"]
    assert result.confidence is not None
    assert result.confidence < 0.5


def test_claim_grounding_deterministic_preserves_negation_contradictions() -> None:
    analyzer = ClaimGroundingAnalyzer()

    result = analyzer._evaluate_claim_deterministic(
        "Warranty covers accidental damage.",
        [chunk("chunk-1", "Warranty does not cover accidental damage.")],
    )

    assert result.label == "contradicted"
    assert result.supporting_chunk_ids == ["chunk-1"]


def test_claim_grounding_evidence_includes_soft_coverage_confidence() -> None:
    run = run_with_answer(
        "Company revenue grew 15% YoY during the fiscal year.",
        [chunk("chunk-1", "Company revenue increased 15% annually during the fiscal year.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)
    claim_result = json.loads(result.evidence[0])

    assert result.status == "pass"
    assert claim_result["label"] == "entailed"
    assert claim_result["supporting_chunk_ids"] == ["chunk-1"]
    assert claim_result["confidence"] >= 0.5
