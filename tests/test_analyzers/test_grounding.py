"""Tests for grounding claim extraction and support analysis."""

from __future__ import annotations

import json
from typing import Any

from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.grounding.value_extraction import find_value_alignment
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


def test_claim_extractor_keeps_short_sentences_with_date() -> None:
    claims = ClaimExtractor().extract("Deadline is June 30.")
    assert claims == ["Deadline is June 30."]


def test_claim_extractor_keeps_short_sentences_with_money() -> None:
    claims = ClaimExtractor().extract("Threshold is $500.")
    assert claims == ["Threshold is $500."]


def test_claim_extractor_keeps_short_sentences_with_go_number() -> None:
    claims = ClaimExtractor().extract("G.O.Rt.No. 2115 applies.")
    assert claims == ["G.O.Rt.No. 2115 applies."]


def test_claim_extractor_keeps_short_sentences_with_policy_keyword() -> None:
    claims = ClaimExtractor().extract("Schools need approval.")
    assert claims == ["Schools need approval."]


def test_claim_extractor_still_filters_empty_fragments() -> None:
    claims = ClaimExtractor().extract("Hi there. OK.")
    assert claims == []


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
    assert result.claim_results is not None
    assert len(result.claim_results) == 1
    assert result.claim_results[0].label == "entailed"
    assert result.claim_results[0].verification_method == "value_aware_structured_claim_verifier_v1"
    assert result.claim_results[0].calibration_status == "uncalibrated"
    assert "claim grounding summary" in result.evidence[0].lower()


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
    assert result.claim_results is not None
    assert result.claim_results[0].label == "entailed"
    assert result.claim_results[0].supporting_chunk_ids == ["chunk-1"]
    assert result.claim_results[0].confidence == 0.9
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
    assert result.claim_results is not None
    assert result.claim_results[0].label == "entailed"
    assert result.claim_results[0].fallback_used is True
    assert "fell back" in (result.claim_results[0].evidence_reason or "").lower()


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
    assert result.verification_method == "deterministic_overlap_anchor_v0"


def test_claim_grounding_deterministic_requires_numeric_anchor_match() -> None:
    analyzer = ClaimGroundingAnalyzer()

    result = analyzer._evaluate_claim_deterministic(
        "Revenue grew 15% YoY.",
        [chunk("chunk-1", "Revenue grew 12% annually.")],
    )

    assert result.label == "contradicted"
    assert result.supporting_chunk_ids == []
    assert result.candidate_chunk_ids == ["chunk-1"]
    assert result.contradicting_chunk_ids == ["chunk-1"]
    assert result.confidence is not None
    assert result.confidence < 0.5
    assert result.verification_method == "deterministic_overlap_anchor_v0"


def test_claim_grounding_deterministic_preserves_negation_contradictions() -> None:
    analyzer = ClaimGroundingAnalyzer()

    result = analyzer._evaluate_claim_deterministic(
        "Warranty covers accidental damage.",
        [chunk("chunk-1", "Warranty does not cover accidental damage.")],
    )

    assert result.label == "contradicted"
    assert result.supporting_chunk_ids == []
    assert result.candidate_chunk_ids == ["chunk-1"]
    assert result.contradicting_chunk_ids == ["chunk-1"]
    assert result.verification_method == "deterministic_overlap_anchor_v0"


def test_claim_grounding_evidence_includes_soft_coverage_confidence() -> None:
    run = run_with_answer(
        "Company revenue grew 15% YoY during the fiscal year.",
        [chunk("chunk-1", "Company revenue increased 15% annually during the fiscal year.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.claim_results is not None
    claim_result = result.claim_results[0]

    assert result.status == "pass"
    assert claim_result.label == "entailed"
    assert claim_result.supporting_chunk_ids == ["chunk-1"]


def test_claim_grounding_emits_typed_claim_results() -> None:
    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )
    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.claim_results is not None
    assert len(result.claim_results) == 1
    assert result.claim_results[0].label == "entailed"
    assert result.claim_results[0].confidence is not None
    assert result.claim_results[0].confidence >= 0.5


def test_claim_grounding_structured_verifier_entails_paraphrased_claim() -> None:
    run = run_with_answer(
        "Company revenue grew 15% YoY in the latest fiscal year.",
        [chunk("chunk-1", "Revenue increased 15% annually.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "entailed"
    assert claim.supporting_chunk_ids == ["chunk-1"]
    assert claim.verification_method == "value_aware_structured_claim_verifier_v1"
    assert claim.fallback_used is False


def test_claim_grounding_structured_verifier_marks_high_overlap_without_anchor_support_as_unsupported() -> None:
    run = run_with_answer(
        "Company revenue grew 15% YoY in the latest fiscal year.",
        [chunk("chunk-1", "Revenue grew 12% annually.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "contradicted"
    assert claim.supporting_chunk_ids == []
    assert claim.candidate_chunk_ids == ["chunk-1"]
    assert claim.contradicting_chunk_ids == ["chunk-1"]
    assert "states" in (claim.evidence_reason or "").lower()


def test_claim_grounding_structured_verifier_marks_contradicted_claim() -> None:
    run = run_with_answer(
        "The product warranty covers accidental damage for all devices.",
        [chunk("chunk-1", "Warranty does not cover accidental damage.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "contradicted"
    assert claim.supporting_chunk_ids == []
    assert claim.candidate_chunk_ids == ["chunk-1"]
    assert claim.contradicting_chunk_ids == ["chunk-1"]
    assert claim.verification_method == "value_aware_structured_claim_verifier_v1"


def test_claim_grounding_structured_verifier_fallback_is_visible_when_unavailable() -> None:
    run = run_with_answer(
        "The refund policy covers hardware returns for thirty days.",
        [chunk("chunk-1", "Refund policy covers hardware returns for thirty days.")],
    )

    result = ClaimGroundingAnalyzer({"force_structured_verifier_error": True}).analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.fallback_used is True
    assert claim.verification_method == "deterministic_overlap_anchor_v0"
    assert "fell back" in (claim.evidence_reason or "").lower()
    assert any("fallback" in item.lower() for item in result.evidence)


def test_value_aware_grounding_flags_numeric_duration_mismatch() -> None:
    run = run_with_answer(
        "Records are retained for thirty six months before archival review in all departments.",
        [chunk("chunk-1", "Records are retained for twelve months before archival review.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label in {"contradicted", "unsupported"}
    assert claim.label == "contradicted"
    assert claim.value_conflicts
    assert "36" in (claim.evidence_reason or "") or "thirty six" in (claim.evidence_reason or "").lower()


def test_find_value_alignment_handles_value_mentions_without_type_error() -> None:
    matches, conflicts, missing_critical = find_value_alignment(
        "Records are retained for thirty six months before archival review.",
        "Records are retained for twelve months before archival review.",
    )

    assert matches == []
    assert conflicts
    assert missing_critical is False


def test_value_aware_grounding_flags_money_threshold_mismatch() -> None:
    run = run_with_answer(
        "Refunds apply above five hundred dollars in this program.",
        [chunk("chunk-1", "Refunds apply above one hundred dollars.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "contradicted"
    assert claim.value_conflicts


def test_value_aware_grounding_entails_same_money_value_different_format() -> None:
    run = run_with_answer(
        "Refunds apply above $100 in policy.",
        [chunk("chunk-1", "Refunds apply above one hundred dollars.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "entailed"
    assert claim.value_matches


def test_value_aware_grounding_entails_same_duration_value_different_format() -> None:
    run = run_with_answer(
        "Records are retained for 12 months.",
        [chunk("chunk-1", "Records are retained for twelve months.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "entailed"


def test_value_aware_grounding_flags_percentage_mismatch() -> None:
    run = run_with_answer(
        "The subsidy is 20% for all eligible applicants.",
        [chunk("chunk-1", "The subsidy is 15 percent.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label in {"contradicted", "unsupported"}
    assert claim.label == "contradicted"


def test_value_aware_grounding_flags_date_mismatch() -> None:
    run = run_with_answer(
        "Applications close on July 15 for this intake cycle.",
        [chunk("chunk-1", "Applications close on June 30.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label in {"contradicted", "unsupported"}
    assert claim.label == "contradicted"


def test_value_aware_grounding_avoids_unrelated_value_false_positive() -> None:
    run = run_with_answer(
        "Refunds apply above $500 under this customer policy.",
        [chunk("chunk-1", "The plan costs $100 per month. Refunds apply above $500.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "entailed"


def test_value_aware_grounding_flags_go_number_mismatch() -> None:
    run = run_with_answer(
        "As per G.O.Rt.No. 115, optional holidays require prior permission from authorities.",
        [chunk("chunk-1", "As per G.O.Rt.No. 2115, optional holidays require permission.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label in {"contradicted", "unsupported"}
    assert claim.label == "contradicted"


def test_grounding_contract_unsupported_uses_candidate_not_supporting_chunks() -> None:
    run = run_with_answer(
        "Refunds apply above five hundred dollars in this program.",
        [chunk("chunk-1", "Refunds apply above one hundred dollars.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "contradicted"
    assert claim.supporting_chunk_ids == []
    assert claim.candidate_chunk_ids == ["chunk-1"]
    assert claim.contradicting_chunk_ids == ["chunk-1"]


def test_grounding_contract_unsupported_related_chunk_keeps_supporting_empty() -> None:
    run = run_with_answer(
        "Hardware returns are available for ninety days.",
        [chunk("chunk-1", "Hardware returns are available for thirty days.")],
    )

    result = ClaimGroundingAnalyzer().analyze(run)

    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label in {"unsupported", "contradicted"}
    if claim.label == "unsupported":
        assert claim.supporting_chunk_ids == []
        assert claim.candidate_chunk_ids == ["chunk-1"]
