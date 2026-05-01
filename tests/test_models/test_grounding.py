"""
Tests for the domain-agnostic grounding models.
"""

import pytest
from raggov.models.grounding import (
    ClaimEvidenceRecord,
    ClaimVerificationLabel,
    CalibrationStatus,
    StructuredClaimRepresentation
)


def test_claim_verification_label_enum():
    assert ClaimVerificationLabel.ENTAILED == "entailed"
    assert ClaimVerificationLabel.CONTRADICTED == "contradicted"
    assert ClaimVerificationLabel.INSUFFICIENT == "insufficient"


def test_calibration_status_enum():
    assert CalibrationStatus.UNCALIBRATED == "uncalibrated"
    assert CalibrationStatus.HEURISTIC == "heuristic"


def test_structured_representation_creation():
    rep = StructuredClaimRepresentation(
        triplet={"subject": "GovRAG", "predicate": "is", "object": "agnostic"},
        qa_pair={"question": "Is GovRAG agnostic?", "answer": "Yes"}
    )
    assert rep.triplet["subject"] == "GovRAG"
    assert rep.qa_pair["answer"] == "Yes"
    assert rep.frame is None


def test_claim_evidence_record_minimal():
    record = ClaimEvidenceRecord(
        claim_id="c1",
        claim_text="GovRAG is a framework."
    )
    assert record.claim_id == "c1"
    assert record.verification_label == ClaimVerificationLabel.UNVERIFIED
    assert record.calibration_status == CalibrationStatus.UNAVAILABLE
    assert record.cited_chunk_ids == []


def test_claim_evidence_record_full():
    record = ClaimEvidenceRecord(
        claim_id="c1",
        claim_text="The subsidy is 60%.",
        source_answer_span=(0, 20),
        verification_label=ClaimVerificationLabel.ENTAILED,
        verifier_method="heuristic_v0",
        verifier_score=0.95,
        calibration_status=CalibrationStatus.HEURISTIC,
        uncertainty_signals={"entropy": 0.1},
        domain_adapter_outputs={"policy_type": "subsidy"},
        structured_representation=StructuredClaimRepresentation(
            triplet={"sub": "subsidy", "pred": "is", "obj": "60%"}
        ),
        metadata={"project": "GovRAG"}
    )
    
    assert record.verification_label == "entailed"
    assert record.verifier_score == 0.95
    assert record.structured_representation.triplet["obj"] == "60%"
    assert record.domain_adapter_outputs["policy_type"] == "subsidy"


def test_claim_evidence_record_serialization():
    record = ClaimEvidenceRecord(
        claim_id="c2",
        claim_text="Test claim",
        verification_label=ClaimVerificationLabel.CONTRADICTED
    )
    
    data = record.model_dump()
    assert data["claim_id"] == "c2"
    assert data["verification_label"] == "contradicted"
    
    # Round trip
    new_record = ClaimEvidenceRecord(**data)
    assert new_record.verification_label == ClaimVerificationLabel.CONTRADICTED


def test_extra_fields_allowed():
    # model_config allows extra
    record = ClaimEvidenceRecord(
        claim_id="c3",
        claim_text="Extra field test",
        some_new_experimental_field="hello"
    )
    assert record.some_new_experimental_field == "hello"
