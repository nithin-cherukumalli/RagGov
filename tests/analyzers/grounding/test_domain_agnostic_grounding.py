import pytest
from unittest.mock import MagicMock

from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.verifiers import (
    ConservativeEnsembleVerifier,
    VerificationResult,
)


def test_software_version_grounding() -> None:
    """Test grounding validation on a software version claim."""
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    # Mock LLM to return supported
    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk_sf",
        evidence_span=None,
        rationale="Matches exact version.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_sf",
            chunk_text="Version 2.4 of the SDK introduces automatic retries for failed requests.",
            source_doc_id="doc_sf",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    # Valid version matches
    res = verifier.verify(
        "The SDK supports retries in version 2.4.",
        "Query",
        candidates,
        metadata={"numbers": ["2.4"], "entities": ["SDK"]},
    )
    assert res.support_label == "supported"

    # Conflicting version should downgrade or fail (e.g. claim 2.5 but chunk says 2.4)
    res_conflict = verifier.verify(
        "The SDK supports retries in version 2.5.",
        "Query",
        candidates,
        metadata={"numbers": ["2.5"], "entities": ["SDK"]},
    )
    # The facts/heuristic gate will detect that "2.5" is missing or conflicting and downgrade
    assert res_conflict.support_label == "contradicted"
    assert res_conflict.safety_gate_triggered is True
    assert res_conflict.safety_gate_reason in {
        "critical_value_missing_or_conflicting",
        "llm_heuristic_disagreement",
    }


def test_healthcare_dosage_grounding() -> None:
    """Test grounding validation on a healthcare dosage limit claim."""
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk_hc",
        evidence_span=None,
        rationale="Matches exact dose.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_hc",
            chunk_text="For adults, the recommended dose of medicine X is 50 mg daily.",
            source_doc_id="doc_hc",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify(
        "The recommended dose of medicine X is 50 mg for adults.",
        "Query",
        candidates,
        metadata={"numbers": ["50"], "entities": ["medicine X", "adults"]},
    )
    assert res.support_label == "supported"

    # Conflicting dosage
    res_conflict = verifier.verify(
        "The recommended dose of medicine X is 100 mg for adults.",
        "Query",
        candidates,
        metadata={"numbers": ["100"], "entities": ["medicine X", "adults"]},
    )
    assert res_conflict.support_label == "contradicted"


def test_finance_rate_grounding() -> None:
    """Test grounding validation on a financial interest rate claim."""
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk_fn",
        evidence_span=None,
        rationale="Matches exact interest rate.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_fn",
            chunk_text="The bank offers a savings account with an annual interest rate of 3.5%.",
            source_doc_id="doc_fn",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify(
        "The annual interest rate on the savings account is 3.5%.",
        "Query",
        candidates,
        metadata={"numbers": ["3.5"], "entities": ["savings account"]},
    )
    assert res.support_label == "supported"


def test_product_manual_spec_grounding_with_decomposition() -> None:
    """Test grounding validation on a compound product manual spec claim with decomposition."""
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    # The claim is compound: "The laptop weighs 1.2 kg and has a battery life of 10 hours."
    # The decomposer will split it into two subclaims:
    # 1. "The laptop weighs 1.2 kg"
    # 2. "has a battery life of 10 hours." (or similar)
    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk_pr",
        evidence_span=None,
        rationale="Matches specification.",
        verifier_name="llm_mock",
    )
    # The base verifier verify() is called twice (once for each subclaim)
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_pr",
            chunk_text="The laptop weighs 1.2 kg and provides up to 10 hours of battery life on a single charge.",
            source_doc_id="doc_pr",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify(
        "The laptop weighs 1.2 kg and has a battery life of 10 hours.",
        "Query",
        candidates,
        metadata={
            "atomicity_status": "compound",
            "numbers": ["1.2", "10"],
            "entities": ["laptop"],
        },
    )
    assert res.support_label == "supported"
    assert res.compound_decomposed is True
    assert len(res.subclaim_results) == 2


def test_scientific_paper_grounding() -> None:
    """Test grounding validation on a scientific trial result claim."""
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk_sc",
        evidence_span=None,
        rationale="Matches exact cohort and CI.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk_sc",
            chunk_text="In this trial, we enrolled 150 patients and calculated a 95% confidence interval.",
            source_doc_id="doc_sc",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify(
        "The trial included 150 patients and showed a 95% confidence interval.",
        "Query",
        candidates,
        metadata={"numbers": ["150", "95"], "entities": ["patients"]},
    )
    assert res.support_label == "supported"
