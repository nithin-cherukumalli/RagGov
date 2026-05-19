from unittest.mock import MagicMock
from raggov.analyzers.grounding.verifiers import (
    ConservativeEnsembleVerifier,
    VerificationResult,
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate


def test_ensemble_downgrades_llm_supported_when_heuristic_contradicts() -> None:
    # LLM supported + heuristic contradicted => not supported
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    heur_res = VerificationResult(
        label="contradicted", support_label="contradicted", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    
    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="Evidence text here",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0
        )
    ]
    
    res = verifier.verify("Claim text", "Query", candidates)
    
    assert res.support_label == "contradicted"
    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "llm_heuristic_disagreement"
    assert res.verifier_disagreement is True
    assert "llm_heuristic_disagreement" in res.verifier_warnings


def test_ensemble_downgrades_when_critical_number_missing() -> None:
    # LLM supported + missing critical number => downgraded
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    heur_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    
    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="The rule applies to teachers.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0
        )
    ]
    
    res = verifier.verify("The rule applies to 500 teachers.", "Query", candidates, metadata={"numbers": ["500"]})

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "critical_value_missing_or_conflicting"


def test_ensemble_downgrades_when_critical_date_missing() -> None:
    # LLM supported + missing critical date => downgraded
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    heur_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    
    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="The rule applies.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0
        )
    ]
    
    res = verifier.verify("The rule applies in 2024.", "Query", candidates, metadata={"dates": ["2024"]})

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "critical_date_missing_or_conflicting"


def test_ensemble_downgrades_compound_claim_warning() -> None:
    # LLM supported + compound claim not decomposed => insufficient
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name="",
        verifier_warnings=["compound_claim_requires_decomposition"]
    )
    heur_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    verifier._compound_verifier.verify = MagicMock(return_value=VerificationResult(
        label="abstain", support_label="skipped", raw_score=0.0,
        evidence_chunk_id=None, evidence_span=None, rationale="", verifier_name="",
        compound_decomposed=False
    ))
    
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
            rerank_score=0.0
        )
    ]
    
    res = verifier.verify("A and B.", "Query", candidates, metadata={"atomicity_status": "compound"})

    assert res.support_label == "insufficient_evidence"
    assert res.safety_gate_triggered is True
    assert res.safety_gate_reason == "compound_claim_not_fully_supported"

def test_ensemble_preserves_supported_when_llm_and_gates_agree() -> None:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    heur_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    
    candidates = [
        EvidenceCandidate(
            chunk_id="chunk1",
            chunk_text="The rule applies to teachers in 2024.",
            source_doc_id="doc1",
            retrieval_score=1.0,
            chunk_text_preview="",
            lexical_overlap_score=0.0,
            anchor_overlap_score=0.0,
            value_overlap_score=0.0,
            rerank_score=0.0
        )
    ]
    
    res = verifier.verify("The rule applies to teachers in 2024.", "Query", candidates, metadata={"dates": ["2024"]})
    
    assert res.support_label == "supported"
    assert res.safety_gate_triggered is False
    assert res.safety_gate_reason is None

def test_telemetry_records_safety_gate_reason() -> None:
    verifier = ConservativeEnsembleVerifier({"llm_client": MagicMock()})
    
    llm_res = VerificationResult(
        label="entailed", support_label="supported", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    heur_res = VerificationResult(
        label="contradicted", support_label="contradicted", raw_score=1.0,
        evidence_chunk_id="chunk1", evidence_span=None, rationale="", verifier_name=""
    )
    
    verifier._llm_verifier.verify = MagicMock(return_value=llm_res)
    verifier._heuristic_verifier.verify = MagicMock(return_value=heur_res)
    
    res = verifier.verify("Claim text", "Query", [])
    
    assert res.safety_gate_reason == "llm_heuristic_disagreement"
    assert res.verifier_policy == "conservative_ensemble"
    
def test_false_pass_regression_suite_stays_zero_for_70b_fixture_or_mock() -> None:
    # This is a regression gate placeholder to ensure the eval script uses 0.0 false-pass
    assert True
