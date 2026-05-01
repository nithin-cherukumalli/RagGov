"""Tests for the v0 heuristic claim-level evidence layer."""

from __future__ import annotations

import pytest

from raggov.analyzers.grounding.evidence_layer import (
    ClaimEvidenceBuilder,
    ClaimEvidenceRecord,
    HeuristicValueOverlapVerifier,
    VerificationOutput,
    detect_atomicity,
    detect_claim_type,
)
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=None,
    )


def verifier(config: dict | None = None) -> HeuristicValueOverlapVerifier:
    return HeuristicValueOverlapVerifier(config or {})


def builder(config: dict | None = None) -> ClaimEvidenceBuilder:
    return ClaimEvidenceBuilder(verifier(config))


# ---------------------------------------------------------------------------
# detect_claim_type
# ---------------------------------------------------------------------------

def test_detect_claim_type_go_number() -> None:
    assert detect_claim_type("As per G.O.Rt.No. 2115, optional holidays apply.") == "go_number"


def test_detect_claim_type_go_number_ms_variant() -> None:
    assert detect_claim_type("G.O.Ms.No. 33 prescribes revised pay scales.") == "go_number"


def test_detect_claim_type_numeric_percentage() -> None:
    assert detect_claim_type("The subsidy is 20% for all eligible applicants.") == "numeric"


def test_detect_claim_type_numeric_currency() -> None:
    assert detect_claim_type("Refunds apply above Rs. 50,000 for this scheme.") == "numeric"


def test_detect_claim_type_date_or_deadline_month() -> None:
    assert detect_claim_type("Applications close on July 15 for this intake cycle.") == "date_or_deadline"


def test_detect_claim_type_date_or_deadline_keyword() -> None:
    assert detect_claim_type("The deadline for submission is next Friday.") == "date_or_deadline"


def test_detect_claim_type_eligibility() -> None:
    assert detect_claim_type("Teachers with 3 years of service are eligible for transfer.") == "eligibility"


def test_detect_claim_type_policy_rule() -> None:
    assert detect_claim_type("All employees must submit a leave form in advance.") == "policy_rule"


def test_detect_claim_type_definition() -> None:
    assert detect_claim_type("Qualifying service means continuous service in a permanent post.") == "definition"


def test_detect_claim_type_general_factual() -> None:
    assert detect_claim_type("The department issued a notice to all staff.") == "general_factual"


def test_detect_claim_type_go_takes_priority_over_numeric() -> None:
    # GO number claims contain numbers; GO type should win
    assert detect_claim_type("G.O.Ms.No. 45 sets the rate at 20%.") == "go_number"


# ---------------------------------------------------------------------------
# detect_atomicity
# ---------------------------------------------------------------------------

def test_detect_atomicity_atomic() -> None:
    claim = "The refund policy covers hardware returns for thirty days."
    assert detect_atomicity(claim) == "atomic"


def test_detect_atomicity_compound_multiple_conjunctions() -> None:
    claim = (
        "The refund policy covers hardware and also applies to software "
        "licenses additionally."
    )
    assert detect_atomicity(claim) == "compound"


def test_detect_atomicity_compound_multiple_verbs() -> None:
    # "covers", "applies", "requires" → 3 finite verbs → compound
    claim = "The policy covers all items, applies to all staff, and requires prior approval."
    assert detect_atomicity(claim) == "compound"


def test_detect_atomicity_unclear_short_claim() -> None:
    assert detect_atomicity("GO applies.") == "unclear"


# ---------------------------------------------------------------------------
# HeuristicValueOverlapVerifier — supported numeric claim
# ---------------------------------------------------------------------------

def test_heuristic_verifier_supported_numeric_claim() -> None:
    output = verifier().verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert output.label == "entailed"
    assert output.raw_support_score >= 0.5
    assert not output.fallback_used
    assert output.calibrated_confidence is None if hasattr(output, "calibrated_confidence") else True


def test_heuristic_verifier_supported_numeric_claim_returns_verification_output() -> None:
    output = verifier().verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert isinstance(output, VerificationOutput)
    assert output.verification_method == "value_aware_structured_claim_verifier_v1"


# ---------------------------------------------------------------------------
# HeuristicValueOverlapVerifier — contradicted numeric claim
# ---------------------------------------------------------------------------

def test_heuristic_verifier_contradicted_numeric_claim() -> None:
    output = verifier().verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue grew 12% annually.")],
    )
    assert output.label == "contradicted"
    assert output.value_conflicts


def test_heuristic_verifier_contradicted_numeric_sets_correct_chunk_ids() -> None:
    output = verifier().verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue grew 12% annually.")],
    )
    assert "c1" in output.candidate_chunk_ids
    assert "c1" in output.contradicting_chunk_ids
    assert output.supporting_chunk_ids == []


# ---------------------------------------------------------------------------
# HeuristicValueOverlapVerifier — unsupported: high lexical overlap, missing value
# ---------------------------------------------------------------------------

def test_heuristic_verifier_unsupported_high_overlap_missing_value() -> None:
    # Claim has Rs. 50,000 but evidence has no numeric value for the refund limit
    output = verifier().verify(
        "The refund limit is Rs. 50,000 for all customers.",
        "refund",
        [chunk("c1", "The refund policy covers all customers in the scheme.")],
    )
    assert output.label == "unsupported"
    assert output.supporting_chunk_ids == []


def test_heuristic_verifier_unsupported_go_number_mismatch() -> None:
    output = verifier().verify(
        "As per G.O.Rt.No. 115, optional holidays require prior permission.",
        "holidays",
        [chunk("c1", "As per G.O.Rt.No. 2115, optional holidays require permission.")],
    )
    assert output.label in {"contradicted", "unsupported"}


# ---------------------------------------------------------------------------
# HeuristicValueOverlapVerifier — date/deadline claim
# ---------------------------------------------------------------------------

def test_heuristic_verifier_date_claim_contradiction() -> None:
    output = verifier().verify(
        "Applications close on July 15 for this intake cycle.",
        "applications",
        [chunk("c1", "Applications close on June 30.")],
    )
    assert output.label in {"contradicted", "unsupported"}


def test_heuristic_verifier_date_claim_supported() -> None:
    output = verifier().verify(
        "The scheme closes on June 30.",
        "scheme",
        [chunk("c1", "Applications and claims under this scheme close on June 30.")],
    )
    assert output.label == "entailed"


# ---------------------------------------------------------------------------
# HeuristicValueOverlapVerifier — fallback_used when structured verifier errors
# ---------------------------------------------------------------------------

def test_heuristic_verifier_fallback_used_when_forced_error() -> None:
    output = verifier({"force_structured_verifier_error": True}).verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert output.fallback_used is True
    assert output.verification_method == "deterministic_overlap_anchor_v0"
    assert "fell back" in output.evidence_reason.lower()


def test_heuristic_verifier_fallback_produces_valid_output() -> None:
    output = verifier({"force_structured_verifier_error": True}).verify(
        "Revenue grew 15% YoY.",
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert output.label in {"entailed", "unsupported", "contradicted"}
    assert output.raw_support_score >= 0.0


# ---------------------------------------------------------------------------
# ClaimEvidenceBuilder
# ---------------------------------------------------------------------------

def test_claim_evidence_builder_returns_records() -> None:
    records = builder().build(
        [
            "Revenue grew 15% YoY.",
            "The policy covers hardware returns for thirty days.",
        ],
        "revenue policy",
        [chunk("c1", "Revenue increased 15% annually. Policy covers hardware returns for 30 days.")],
    )
    assert len(records) == 2
    assert all(isinstance(r, ClaimEvidenceRecord) for r in records)


def test_claim_evidence_builder_assigns_sequential_claim_ids() -> None:
    records = builder().build(
        ["Claim one here.", "Claim two here too."],
        "claims",
        [chunk("c1", "Claim one here. Claim two here too.")],
    )
    assert records[0].claim_id == "claim_001"
    assert records[1].claim_id == "claim_002"


def test_claim_evidence_builder_calibrated_confidence_is_none() -> None:
    records = builder().build(
        ["Revenue grew 15% YoY."],
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert records[0].calibrated_confidence is None
    assert records[0].calibration_status == "uncalibrated"


def test_claim_evidence_builder_sets_claim_type() -> None:
    records = builder().build(
        ["The subsidy is 20% for all applicants."],
        "subsidy",
        [chunk("c1", "The subsidy rate is 20%.")],
    )
    assert records[0].claim_type == "numeric"


def test_claim_evidence_builder_sets_atomicity_status() -> None:
    records = builder().build(
        ["Revenue grew 15% YoY and also applies to all fiscal years additionally."],
        "revenue",
        [chunk("c1", "Revenue grew 15% annually.")],
    )
    assert records[0].atomicity_status == "compound"


def test_claim_evidence_builder_extracts_values() -> None:
    records = builder().build(
        ["The refund limit is Rs. 50,000."],
        "refund",
        [chunk("c1", "The refund limit is Rs. 50,000.")],
    )
    assert records[0].extracted_values is not None


def test_claim_evidence_builder_fallback_propagates_to_record() -> None:
    records = builder({"force_structured_verifier_error": True}).build(
        ["Revenue grew 15% YoY."],
        "revenue",
        [chunk("c1", "Revenue increased 15% annually.")],
    )
    assert records[0].fallback_used is True
    assert records[0].calibrated_confidence is None


# ---------------------------------------------------------------------------
# Backward compatibility — ClaimGroundingAnalyzer output unchanged
# ---------------------------------------------------------------------------

def test_backward_compat_analyzer_passes_on_entailed_claim() -> None:
    run = RAGRun(
        query="coverage",
        retrieved_chunks=[chunk("c1", "Refund policy covers hardware returns for thirty days.")],
        final_answer="The refund policy covers hardware returns for thirty days.",
    )
    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.status == "pass"
    assert result.claim_results is not None
    assert result.claim_results[0].label == "entailed"
    assert result.claim_results[0].calibration_status == "uncalibrated"
    assert "claim grounding summary" in result.evidence[0].lower()


def test_backward_compat_analyzer_warns_on_contradicted_claim() -> None:
    run = RAGRun(
        query="warranty",
        retrieved_chunks=[chunk("c1", "Warranty does not cover accidental damage.")],
        final_answer="The product warranty covers accidental damage for all devices.",
    )
    result = ClaimGroundingAnalyzer({"fail_threshold": 1.1}).analyze(run)
    assert result.status == "warn"
    assert result.claim_results is not None
    assert result.claim_results[0].label == "contradicted"


def test_backward_compat_fallback_used_visible_in_claim_result() -> None:
    run = RAGRun(
        query="returns",
        retrieved_chunks=[chunk("c1", "Refund policy covers hardware returns for thirty days.")],
        final_answer="The refund policy covers hardware returns for thirty days.",
    )
    result = ClaimGroundingAnalyzer({"force_structured_verifier_error": True}).analyze(run)
    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.fallback_used is True
    assert claim.verification_method == "deterministic_overlap_anchor_v0"
    assert "fell back" in (claim.evidence_reason or "").lower()


def test_backward_compat_calibration_note_in_evidence_reason() -> None:
    run = RAGRun(
        query="coverage",
        retrieved_chunks=[chunk("c1", "Refund policy covers hardware returns for thirty days.")],
        final_answer="The refund policy covers hardware returns for thirty days.",
    )
    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.claim_results is not None
    reason = result.claim_results[0].evidence_reason or ""
    assert "uncalibrated" in reason.lower()


def test_backward_compat_value_conflicts_propagate() -> None:
    run = RAGRun(
        query="revenue",
        retrieved_chunks=[chunk("c1", "Revenue grew 12% annually.")],
        final_answer="Company revenue grew 15% YoY in the latest fiscal year.",
    )
    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "contradicted"
    assert claim.value_conflicts


def test_backward_compat_supporting_chunk_ids_populated_for_entailed() -> None:
    run = RAGRun(
        query="revenue",
        retrieved_chunks=[chunk("c1", "Revenue increased 15% annually.")],
        final_answer="Company revenue grew 15% YoY in the latest fiscal year.",
    )
    result = ClaimGroundingAnalyzer().analyze(run)
    assert result.claim_results is not None
    claim = result.claim_results[0]
    assert claim.label == "entailed"
    assert "c1" in claim.supporting_chunk_ids
