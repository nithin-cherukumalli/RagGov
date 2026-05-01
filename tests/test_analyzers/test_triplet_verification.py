"""
Tests for triplet-level verification and claim aggregation.
"""

import pytest
from unittest.mock import MagicMock
from raggov.analyzers.grounding.evidence_layer import (
    ClaimEvidenceBuilder,
    ClaimEvidenceRecord
)
from raggov.analyzers.grounding.verifiers import (
    TripletVerificationResult,
    VerificationResult,
    HeuristicValueOverlapVerifier,
    TripletVerifier
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.triplets import ClaimTriplet


class MockTripletVerifier(TripletVerifier):
    def __init__(self, results):
        self.results = results
        self.call_count = 0

    def verify_triplets(self, triplets, candidates, metadata=None):
        self.call_count += 1
        return self.results


def test_aggregation_all_entailed():
    builder = ClaimEvidenceBuilder(MagicMock(), MagicMock())
    triplet_results = [
        TripletVerificationResult(label="entailed", raw_score=0.9, method="mock"),
        TripletVerificationResult(label="entailed", raw_score=0.8, method="mock")
    ]
    base = VerificationResult(label="unsupported", raw_score=0.0, evidence_chunk_id=None, evidence_span=None, rationale="", verifier_name="base")
    
    aggregated = builder._aggregate_triplet_results(triplet_results, base)
    assert aggregated.label == "entailed"
    assert aggregated.raw_score == pytest.approx(0.85)


def test_aggregation_one_contradicted():
    builder = ClaimEvidenceBuilder(MagicMock(), MagicMock())
    triplet_results = [
        TripletVerificationResult(label="entailed", raw_score=0.9, method="mock"),
        TripletVerificationResult(label="contradicted", raw_score=0.9, method="mock")
    ]
    base = VerificationResult(label="entailed", raw_score=0.9, evidence_chunk_id=None, evidence_span=None, rationale="", verifier_name="base")
    
    aggregated = builder._aggregate_triplet_results(triplet_results, base)
    assert aggregated.label == "contradicted"


def test_aggregation_one_unsupported():
    builder = ClaimEvidenceBuilder(MagicMock(), MagicMock())
    triplet_results = [
        TripletVerificationResult(label="entailed", raw_score=0.9, method="mock"),
        TripletVerificationResult(label="unsupported", raw_score=0.5, method="mock")
    ]
    base = VerificationResult(label="entailed", raw_score=0.9, evidence_chunk_id=None, evidence_span=None, rationale="", verifier_name="base")
    
    aggregated = builder._aggregate_triplet_results(triplet_results, base)
    assert aggregated.label == "unsupported"


def test_triplet_verification_flow():
    # Setup mocks
    verifier = MagicMock()
    verifier.verify.return_value = VerificationResult(label="unsupported", raw_score=0.0, evidence_chunk_id=None, evidence_span=None, rationale="", verifier_name="base")
    
    selector = MagicMock()
    selector.select_candidates.return_value = [
        EvidenceCandidate(
            chunk_id="c1",
            source_doc_id="d1",
            chunk_text="Evidence text",
            chunk_text_preview="Evidence text",
            lexical_overlap_score=1.0,
            anchor_overlap_score=1.0,
            value_overlap_score=1.0,
            retrieval_score=1.0,
            rerank_score=None
        )
    ]
    
    extractor = MagicMock()
    extractor.extract.return_value = [ClaimTriplet(triplet_id="t1", source_claim_id="c1", subject="S", predicate="P", object="O")]
    
    triplet_verifier = MockTripletVerifier([
        TripletVerificationResult(label="entailed", raw_score=1.0, supporting_chunk_id="c1", method="triplet_llm")
    ])
    
    builder = ClaimEvidenceBuilder(verifier, selector, triplet_extractor=extractor)
    builder.set_triplet_verifier(triplet_verifier)
    
    # Run
    records = builder.build(["Test claim"], "query", [])
    
    assert len(records) == 1
    assert records[0].verification_label == "entailed"
    assert records[0].verification_method == "triplet_llm"
    assert triplet_verifier.call_count == 1


def test_triplet_verification_fallback_on_extraction_failure():
    verifier = MagicMock()
    verifier.verify.return_value = VerificationResult(label="entailed", raw_score=0.9, evidence_chunk_id="c1", evidence_span=None, rationale="Base rationale", verifier_name="base")
    
    selector = MagicMock()
    selector.select_candidates.return_value = [
        EvidenceCandidate(
            chunk_id="c1",
            source_doc_id="d1",
            chunk_text="E",
            chunk_text_preview="E",
            lexical_overlap_score=0.9,
            anchor_overlap_score=0.9,
            value_overlap_score=0.0,
            retrieval_score=0.9,
            rerank_score=None
        )
    ]
    
    extractor = MagicMock()
    extractor.extract.side_effect = Exception("Extraction failed")
    
    triplet_verifier = MockTripletVerifier([])
    
    builder = ClaimEvidenceBuilder(verifier, selector, triplet_extractor=extractor)
    builder.set_triplet_verifier(triplet_verifier)
    
    records = builder.build(["Test claim"], "q", [])
    
    assert records[0].verification_label == "entailed"
    assert records[0].verification_method == "base"
    assert triplet_verifier.call_count == 0


def test_aggregation_rationale_concatenation():
    builder = ClaimEvidenceBuilder(MagicMock(), MagicMock())
    triplet_results = [
        TripletVerificationResult(label="entailed", raw_score=0.9, rationale="Reason 1", triplet_id="T1"),
        TripletVerificationResult(label="unsupported", raw_score=0.5, rationale="Reason 2", triplet_id="T2")
    ]
    aggregated = builder._aggregate_triplet_results(triplet_results, MagicMock())
    assert "Reason 1" in aggregated.rationale
    assert "Reason 2" in aggregated.rationale
    assert "T1" in aggregated.rationale
