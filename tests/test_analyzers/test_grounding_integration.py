import pytest
from unittest.mock import MagicMock
from raggov.models.diagnosis import AnalyzerResult, FailureType, ClaimResult
from raggov.models.grounding import GroundingEvidenceBundle, ClaimEvidenceRecord
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.models.run import RAGRun
from raggov.analyzers.taxonomy_classifier.layer6 import Layer6TaxonomyClassifier
from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer

@pytest.fixture
def mock_rag_run():
    run = MagicMock(spec=RAGRun)
    run.final_answer = "The interest rate is 5%."
    run.retrieved_chunks = []
    run.query = "What is the interest rate?"
    return run

@pytest.fixture
def sample_bundle():
    # EvidenceCandidate is a dataclass, use MagicMock for simplicity in test
    candidate = MagicMock()
    candidate.chunk_id = "CH1"
    candidate.lexical_overlap_score = 0.8
    candidate.chunk_text = "Rate is 4%"
    candidate.score = 0.8 # Some downstream code might expect .score
    
    record = ClaimEvidenceRecord(
        claim_id="C1",
        claim_text="The interest rate is 5%",
        verification_label="insufficient",
        candidate_evidence_chunks=[candidate],
        uncertainty_signals={"value_conflicts": ["5% vs 4%"]},
        verifier_score=0.2,
        calibration_status="calibrated"
    )
    return GroundingEvidenceBundle(
        claim_evidence_records=[record],
        diagnostic_rollup={
            "context_ignored_suspected_count": 1,
            "retrieval_miss_suspected_count": 0
        }
    )

def test_layer6_consumes_bundle(mock_rag_run, sample_bundle):
    classifier = Layer6TaxonomyClassifier()
    
    # Prior result with bundle
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        grounding_evidence_bundle=sample_bundle
    )
    
    classifier.config["prior_results"] = [prior_result]
    result = classifier.analyze(mock_rag_run)
    
    # Verify classification
    # Based on diagnostic_rollup, it should identify context_ignored
    evidence_str = "".join(result.evidence)
    assert "context_ignored" in evidence_str or any(f.failure.failure_mode == "context_ignored" for f in result.taxonomy_candidates or [])

def test_a2p_consumes_bundle(mock_rag_run, sample_bundle):
    analyzer = A2PAttributionAnalyzer()
    
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        grounding_evidence_bundle=sample_bundle,
        claim_results=[
            ClaimResult(
                claim_text="The interest rate is 5%",
                label="unsupported",
                confidence=0.2
            )
        ]
    )
    
    analyzer.config["prior_results"] = [prior_result]
    result = analyzer.analyze(mock_rag_run)
    
    # Verify attribution
    # It should detect value_distortion from the record
    primary_attribution = result.claim_attributions[0]
    assert primary_attribution.primary_cause == "value_distortion"
    assert "value conflicts" in primary_attribution.abduct

def test_analyzers_report_missing_bundle(mock_rag_run):
    analyzer = A2PAttributionAnalyzer()
    
    # Prior result WITHOUT bundle
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        claim_results=[
            ClaimResult(
                claim_text="Legacy claim",
                label="unsupported"
            )
        ]
    )
    
    analyzer.config["prior_results"] = [prior_result]
    result = analyzer.analyze(mock_rag_run)
    
    # Should fallback to legacy behavior (weak_or_ambiguous_evidence)
    primary_attribution = result.claim_attributions[0]
    assert primary_attribution.primary_cause == "weak_or_ambiguous_evidence"
