"""Grounded-clean gate (entailment verdict overrides low-tier retrieval-health heuristic).

Phase 2 increment 3 introduced the gate (strict-zero-unsupported). Increment 4 (prereg v2)
loosens condition 3 to an entailed-fraction rule, keeping a hard floor on contradictions and
requiring real positive grounding evidence. Native mode stays byte-identical (the gate needs an
entailment-grade verifier to fire).
"""
from __future__ import annotations

import pytest

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


def test_all_entailed_overrides_stale() -> None:
    results = [_grounding(["entailed", "entailed", "abstain"])]  # abstain excluded from denom
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is True


def test_entailed_fraction_above_threshold_overrides() -> None:
    # 3 entailed / 4 verifiable = 0.75 >= default 0.75 -> suppress (increment-4 loosening)
    results = [_grounding(["entailed", "entailed", "entailed", "unsupported"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is True


def test_entailed_fraction_below_threshold_blocks() -> None:
    # 1 entailed / 2 verifiable = 0.5 < 0.75 -> not suppressed
    results = [_grounding(["entailed", "unsupported"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False


def test_contradiction_is_hard_floor() -> None:
    # Even an otherwise high entailed fraction must NOT be suppressed if any claim is contradicted.
    results = [_grounding(["entailed", "entailed", "entailed", "contradicted"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False


def test_all_abstain_does_not_override() -> None:
    # No verifiable claim -> no positive grounding evidence -> do not suppress (tightening).
    # ClaimResult.label is a 4-value literal; "abstain" is the only abstain-family value.
    results = [_grounding(["abstain", "abstain", "abstain"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False


def test_lexical_grounding_does_not_override() -> None:
    # native heuristic verifier -> not entailment-grade -> gate inactive (no native change)
    results = [_grounding(["entailed", "entailed"], method="deterministic_overlap_anchor_v0")]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False


def test_env_override_relaxes_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 entailed / 2 verifiable = 0.5; default 0.75 blocks, but env override of 0.5 admits it.
    results = [_grounding(["entailed", "unsupported"])]
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is False
    monkeypatch.setenv("RAGGOV_GROUNDED_CLEAN_ENTAILED_FRACTION", "0.5")
    assert grounded_clean_override(_stale_winner(), [_stale_winner()], results) is True
