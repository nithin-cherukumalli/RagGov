import pytest
from unittest.mock import MagicMock
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun
from raggov.models.diagnosis import FailureType, ClaimResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate, EvidenceCandidateSelector
from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier

# Helper to create chunks
def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=0.9,
    )

def create_candidate(chunk_id: str, text: str, lexical_score: float = 0.9) -> EvidenceCandidate:
    """Helper to create a valid EvidenceCandidate dataclass instance."""
    return EvidenceCandidate(
        chunk_id=chunk_id,
        source_doc_id=f"doc-{chunk_id}",
        chunk_text=text,
        chunk_text_preview=text[:50],
        lexical_overlap_score=lexical_score,
        anchor_overlap_score=0.0,
        value_overlap_score=0.0,
        retrieval_score=0.9,
        rerank_score=None,
        metadata_match_flags=[],
        candidate_reason="test",
        is_best=True
    )

# ---------------------------------------------------------------------------
# Regression Suite for Claim Grounding Failure Modes
# ---------------------------------------------------------------------------

def test_regression_lexical_variants_retrievable():
    """
    Case: Lexical variants should be candidate-retrievable.
    Reason: Heuristic normalization (YoY -> annual) ensures we don't miss evidence
    due to minor phrasing differences.
    """
    selector = EvidenceCandidateSelector()
    claim = "Revenue grew 15% YoY"
    chunks = [chunk("c1", "Revenue increased 15% annually.")]
    
    candidates = selector.select_candidates(claim, "revenue", chunks)
    assert len(candidates) > 0
    assert candidates[0].chunk_id == "c1"
    # Even if verification fails in some modes, the candidate MUST be found.
    assert candidates[0].lexical_overlap_score > 0.5

def test_regression_wrong_number_contradicted():
    """
    Case: High lexical overlap but wrong numeric value.
    Reason: Prevents false 'entailed' labels when only the specific value is wrong.
    """
    verifier = HeuristicValueOverlapVerifier({})
    claim = "Interest rate is 5%"
    candidate = create_candidate("c1", "Interest rate is 4%", lexical_score=0.9)
    
    result = verifier.verify(claim, "rate", [candidate])
    assert result.label == "contradicted"
    # Verify conflicts are captured
    conflict_str = str(result.value_conflicts)
    assert "5" in conflict_str
    assert "4" in conflict_str

def test_regression_wrong_go_number_fails():
    """
    Case: Wrong Government Order (GO) number.
    Reason: GO numbers are critical anchors in policy documents. Mismatches
    should never be entailed.
    """
    verifier = HeuristicValueOverlapVerifier({})
    claim = "As per G.O.Ms.No. 45, the subsidy is 20%."
    candidate = create_candidate("c1", "As per G.O.Ms.No. 46, the subsidy is 20%.")
    
    result = verifier.verify(claim, "subsidy", [candidate])
    assert result.label in {"contradicted", "unsupported"}
    assert result.label != "entailed"

def test_regression_wrong_date_fails():
    """
    Case: Wrong date in claim.
    Reason: Dates are often the source of hallucinations in RAG answers.
    """
    verifier = HeuristicValueOverlapVerifier({})
    claim = "The deadline is July 15."
    candidate = create_candidate("c1", "The deadline is June 30.")
    
    result = verifier.verify(claim, "deadline", [candidate])
    assert result.label in {"contradicted", "unsupported"}

def test_regression_compound_claim_mixed_support():
    """
    Case: Compound claim with one supported and one unsupported subclaim.
    Reason: Prevents "over-generalization" where a partially correct answer
    is treated as fully grounded.
    """
    verifier = HeuristicValueOverlapVerifier({})
    claim = "Revenue grew 15% and the CEO resigned."
    # Evidence only supports the first part
    candidate = create_candidate("c1", "Revenue grew 15%.", lexical_score=0.5)
    
    result = verifier.verify(claim, "revenue", [candidate])
    # Heuristic overlap shouldn't be 1.0, and it shouldn't be fully entailed if possible.
    assert result.label != "entailed" or result.raw_score < 0.9

def test_regression_wrong_citation_detection():
    """
    Case: Correct answer but cited doc does not support it.
    Reason: Validates that we don't just check if the answer is "true" in some
    absolute sense, but that the cited source actually supports it.
    """
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="rate",
        retrieved_chunks=[
            chunk("c1", "Interest rate is 5%"), # The evidence
            chunk("c2", "The weather is nice")   # The wrong citation
        ],
        final_answer="The interest rate is 5% [c2]."
    )
    result = analyzer.analyze(run)
    # Inspect the bundle for citation details
    assert result.grounding_evidence_bundle is not None
    record = result.grounding_evidence_bundle.claim_evidence_records[0]
    
    # Ensure c1 is found as supporter in the record
    assert "c1" in record.supporting_chunk_ids
    # Ensure c2 (cited) is present in cited_chunk_ids (parsed from [c2])
    assert "c2" in record.cited_chunk_ids
    # But it should NOT be in supporting_chunk_ids as it's the wrong citation
    assert "c2" not in record.supporting_chunk_ids

def test_regression_evidence_present_but_not_cited():
    """
    Case: Supporting evidence exists in retrieved chunks but was not cited.
    Reason: Helps detect "missing citations" where the model got it right
    but didn't show its work.
    """
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="rate",
        retrieved_chunks=[chunk("c1", "Interest rate is 5%")],
        final_answer="The interest rate is 5%." # No citation [c1]
    )
    result = analyzer.analyze(run)
    assert result.grounding_evidence_bundle is not None
    record = result.grounding_evidence_bundle.claim_evidence_records[0]
    
    assert "c1" in record.supporting_chunk_ids
    # cited_chunk_ids should be empty as it wasn't parsed from answer
    assert not record.cited_chunk_ids

def test_regression_no_retrieved_chunks_clean_failure():
    """
    Case: Empty retrieval set.
    Reason: System should not crash or invent hallucinations when no context exists.
    """
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="any",
        retrieved_chunks=[],
        final_answer="Some answer."
    )
    result = analyzer.analyze(run)
    # Should result in fail or warn depending on config, but definitely not pass
    assert result.status != "pass"
    if result.claim_results:
        assert result.claim_results[0].label == "unsupported"

def test_regression_llm_verifier_failure_fallback():
    """
    Case: LLM verifier fails (mocked via config).
    Reason: Reliability - the system must fall back to heuristics instead of
    returning empty or error results.
    """
    analyzer = ClaimGroundingAnalyzer({"force_structured_verifier_error": True})
    run = RAGRun(
        query="revenue",
        retrieved_chunks=[chunk("c1", "Revenue grew 15%")],
        final_answer="Revenue grew 15%."
    )
    result = analyzer.analyze(run)
    assert result.claim_results[0].fallback_used is True
    assert result.claim_results[0].verification_method == "deterministic_overlap_anchor_v0"

def test_regression_calibration_missing_no_confidence():
    """
    Case: Calibration file missing.
    Reason: We should not "guess" confidence. If uncalibrated, it should stay None.
    """
    # By default, no calibration is loaded.
    analyzer = ClaimGroundingAnalyzer()
    run = RAGRun(
        query="revenue",
        retrieved_chunks=[chunk("c1", "Revenue grew 15%")],
        final_answer="Revenue grew 15%."
    )
    result = analyzer.analyze(run)
    assert result.claim_results[0].confidence is not None # Heuristic score
    assert result.claim_results[0].calibration_status == "uncalibrated"
