"""Phase 2: an entailment-grade contradiction counts as explicit; lexical does not."""
from __future__ import annotations

from raggov.decision_policy import DecisionCandidate, EvidenceTier
from raggov.decision_policy_support import _has_explicit_contradiction
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
)


def _candidate() -> DecisionCandidate:
    return DecisionCandidate(
        failure_type=FailureType.CONTRADICTED_CLAIM,
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        stage=FailureStage.GROUNDING,
        evidence_tier=EvidenceTier.STRUCTURED_DIAGNOSTIC,
        weight=1.0,
        original_index=0,
        calibration_status="uncalibrated",
        recommended_for_gating=False,
        evidence_summary="",
        reason="",
    )


def _result(verification_method: str) -> AnalyzerResult:
    claim = ClaimResult(
        claim_text="The treaty was ratified in 1990.",
        label="contradicted",
        support_label="contradicted",
        verification_method=verification_method,
        label_reason="explicit_contradiction",
    )
    return AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.CONTRADICTED_CLAIM,
        stage=FailureStage.GROUNDING,
        claim_results=[claim],
    )


def test_entailment_method_contradiction_is_explicit() -> None:
    assert _has_explicit_contradiction(
        _candidate(), [_result("llm_claim_entailment_verifier_v1")]
    ) is True


def test_lexical_method_contradiction_is_not_explicit() -> None:
    # Native heuristic verifier — must stay demoted (Task 22 preserved).
    assert _has_explicit_contradiction(
        _candidate(), [_result("deterministic_overlap_anchor_v0")]
    ) is False
