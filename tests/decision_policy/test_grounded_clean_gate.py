"""Phase 2 increment 3: grounded-clean gate (entailment verdict overrides low-tier heuristic)."""
from __future__ import annotations

from raggov.decision_policy import DecisionCandidate, EvidenceTier
from raggov.decision_policy_support import grounded_clean_override
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType


def _stale_winner() -> DecisionCandidate:
    return DecisionCandidate(
        failure_type=FailureType.STALE_RETRIEVAL,
        analyzer_name="StaleRetrievalAnalyzer",
        status="fail",
        stage=FailureStage.RETRIEVAL,
        evidence_tier=EvidenceTier.HEURISTIC_SUPPORTING,
        weight=0.5,
        original_index=0,
        calibration_status="uncalibrated",
        recommended_for_gating=False,
        evidence_summary="",
        reason="",
    )


def _grounding(claim_labels, method="llm_claim_entailment_verifier_v1") -> AnalyzerResult:
    claims = [
        ClaimResult(claim_text=f"c{i}", label=lbl, support_label="supported",
                    verification_method=method)
        for i, lbl in enumerate(claim_labels)
    ]
    return AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer", status="pass",
        stage=FailureStage.GROUNDING, claim_results=claims,
    )


def test_entailment_clean_overrides_stale() -> None:
    results = [_grounding(["entailed", "entailed", "abstain"])]  # index 1
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is True


def test_unsupported_claim_blocks_override() -> None:
    results = [_grounding(["entailed", "unsupported"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False


def test_lexical_grounding_does_not_override() -> None:
    # native heuristic verifier -> not entailment-grade -> gate inactive (no native change)
    results = [_grounding(["entailed", "entailed"], method="deterministic_overlap_anchor_v0")]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False
