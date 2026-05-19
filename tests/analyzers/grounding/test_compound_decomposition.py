import unittest
from unittest.mock import MagicMock

from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.verifiers import (
    ConservativeEnsembleVerifier,
    CompoundClaimVerifier,
    VerificationResult,
)
from raggov.analyzers.grounding.claims import ExtractedClaim


def test_compound_claim_all_parts_supported_parent_supported() -> None:
    # Set up ensemble with a mocked LLM client so it initializes the decomposer.
    # We will mock the base LLM verifier to support all subclaims.
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    # "A and B." will split into two subclaims: "A." and "B." via HeuristicConjunctionSplitter.
    # We want both subclaims to return support_label="supported".
    mock_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=0.9,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="Subclaim is verified.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=mock_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="A and B.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify("A and B.", "Query", candidates, metadata={"atomicity_status": "compound"})
    
    # Assertions
    assert res.support_label == "supported"
    assert res.compound_decomposed is True
    assert len(res.subclaim_results) == 2
    assert res.subclaim_results[0]["support_label"] == "supported"
    assert res.subclaim_results[1]["support_label"] == "supported"
    assert res.undecomposed_compound_gate_triggered is False


def test_compound_claim_one_missing_parent_insufficient() -> None:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    # The base LLM verifier will return "supported" for the first, "insufficient_evidence" for the second.
    res_supported = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=0.9,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="Supported.",
        verifier_name="llm_mock",
    )
    res_insufficient = VerificationResult(
        label="unsupported",
        support_label="insufficient_evidence",
        raw_score=0.0,
        evidence_chunk_id=None,
        evidence_span=None,
        rationale="Missing evidence.",
        verifier_name="llm_mock",
    )

    verifier._llm_verifier.verify = MagicMock(side_effect=[res_supported, res_insufficient])

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="A.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify("A and B.", "Query", candidates, metadata={"atomicity_status": "compound"})

    # Assertions
    assert res.support_label == "insufficient_evidence"
    assert res.compound_decomposed is True
    assert len(res.subclaim_results) == 2
    assert res.subclaim_results[0]["support_label"] == "supported"
    assert res.subclaim_results[1]["support_label"] == "insufficient_evidence"


def test_compound_claim_one_contradicted_parent_contradicted() -> None:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    # First subclaim supported, second contradicted.
    res_supported = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=0.9,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="Supported.",
        verifier_name="llm_mock",
    )
    res_contradicted = VerificationResult(
        label="contradicted",
        support_label="contradicted",
        raw_score=0.0,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="Contradicted.",
        verifier_name="llm_mock",
    )

    verifier._llm_verifier.verify = MagicMock(side_effect=[res_supported, res_contradicted])

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="A.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify("A and B.", "Query", candidates, metadata={"atomicity_status": "compound"})

    # Assertions
    assert res.support_label == "contradicted"
    assert res.compound_decomposed is True
    assert len(res.subclaim_results) == 2
    assert res.subclaim_results[0]["support_label"] == "supported"
    assert res.subclaim_results[1]["support_label"] == "contradicted"


def test_decomposition_failure_triggers_safety_gate() -> None:
    # Create ensemble.
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    # A single sentence "A." with atomicity_status="compound" will fail to yield multiple subclaims.
    # The verifier should return a result where undecomposed_compound_gate_triggered=True
    # and fall through to the old compound safety gate (getting downgraded to insufficient_evidence).
    llm_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="",
        verifier_name="llm_mock",
        verifier_warnings=["compound_claim_requires_decomposition"],
    )
    heur_res = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=1.0,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="",
        verifier_name="heur_mock",
    )

    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="A.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify("A.", "Query", candidates, metadata={"atomicity_status": "compound"})

    # Assertions
    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "compound_claim_not_fully_supported"


def test_subclaim_results_serialized() -> None:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})

    res_supported = VerificationResult(
        label="entailed",
        support_label="supported",
        raw_score=0.9,
        evidence_chunk_id="chunk1",
        evidence_span=None,
        rationale="Supported part.",
        verifier_name="llm_mock",
    )
    verifier._llm_verifier.verify = MagicMock(return_value=res_supported)

    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="A and B.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0,
        )
    ]

    res = verifier.verify("A and B.", "Query", candidates, metadata={"atomicity_status": "compound"})

    # Ensure everything is in subclaim_results serialized list
    assert len(res.subclaim_results) == 2
    for item in res.subclaim_results:
        assert "subclaim_id" in item
        assert "text" in item
        assert "required_support" in item
        assert "label" in item
        assert "support_label" in item
        assert "rationale" in item
        assert "best_candidate_id" in item
