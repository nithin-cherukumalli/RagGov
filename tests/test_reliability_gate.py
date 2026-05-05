import pytest
from raggov.engine import DiagnosisEngine
from raggov.models.run import RAGRun, RetrievedChunk
from raggov.models.diagnosis import FailureType
from raggov.analyzers.base import BaseAnalyzer, AnalyzerResult

class MockPassAnalyzer(BaseAnalyzer):
    def __init__(self, name="MockPassAnalyzer"):
        super().__init__()
        self._name = name
    def name(self): return self._name
    def analyze(self, run):
        return AnalyzerResult(analyzer_name=self._name, status="pass")

class MockSkipAnalyzer(BaseAnalyzer):
    def __init__(self, name="MockSkipAnalyzer"):
        super().__init__()
        self._name = name
    def name(self): return self._name
    def analyze(self, run):
        return AnalyzerResult(analyzer_name=self._name, status="skip")

class MockCrashAnalyzer(BaseAnalyzer):
    def __init__(self, name="MockCrashAnalyzer"):
        super().__init__()
        self._name = name
    def name(self): return self._name
    def analyze(self, run):
        raise ValueError("Simulated crash")

@pytest.fixture
def basic_run():
    return RAGRun(
        run_id="test-run",
        query="What is GovRAG?",
        retrieved_chunks=[RetrievedChunk(chunk_id="c1", text="GovRAG is a tool.", source_doc_id="doc1", score=0.9)],
        final_answer="GovRAG is a tool.",
    )

def test_missing_critical_grounding_prevents_clean(basic_run):
    # external-enhanced mode by default
    # If ClaimGroundingAnalyzer is skipped, status should be INCOMPLETE_DIAGNOSIS
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    # We replace the default ClaimGroundingAnalyzer with a skipping one
    engine.analyzers = [
        MockSkipAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockPassAnalyzer("NCVPipelineVerifier")
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS
    assert "ClaimGroundingAnalyzer" in diagnosis.evidence

def test_missing_ncv_in_enhanced_prevents_clean(basic_run):
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockSkipAnalyzer("NCVPipelineVerifier")
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS
    assert "NCVPipelineVerifier" in diagnosis.evidence

def test_crashing_critical_analyzer_prevents_clean(basic_run):
    # NCVPipelineVerifier is in CRITICAL_ANALYZER_TYPES in engine.py
    from raggov.analyzers.verification.ncv import NCVPipelineVerifier
    
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    
    class CrashingNCV(NCVPipelineVerifier):
        def analyze(self, run): raise ValueError("NCV Crash")
        def name(self): return "NCVPipelineVerifier" # Match critical name
        
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        CrashingNCV({})
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS
    # Check that any evidence contains the analyzer name or it's in the special missing list
    assert any("NCVPipelineVerifier" in e for e in diagnosis.evidence)

def test_native_mode_reliability_gate(basic_run):
    # In native mode, NCVPipelineVerifier is NOT critical by default
    engine = DiagnosisEngine(config={"mode": "native"})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockSkipAnalyzer("NCVPipelineVerifier") # Not critical in native
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.CLEAN

def test_native_mode_missing_grounding_fails(basic_run):
    engine = DiagnosisEngine(config={"mode": "native"})
    engine.analyzers = [
        MockSkipAnalyzer("ClaimGroundingAnalyzer")
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS

def test_native_mode_requires_citation_faithfulness_when_claim_citation_records_exist():
    run_with_claim_citations = RAGRun(
        run_id="native-citation-run",
        query="Q",
        retrieved_chunks=[RetrievedChunk(chunk_id="c1", text="T", source_doc_id="d1", score=1.0)],
        final_answer="A [1].",
        cited_doc_ids=["d1"],
        metadata={"claim_evidence_records": [{"claim_id": "claim-1"}]},
    )
    engine = DiagnosisEngine(config={"mode": "native"})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockSkipAnalyzer("CitationFaithfulnessAnalyzerV0"),
    ]

    diagnosis = engine.diagnose(run_with_claim_citations)

    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS
    assert "CitationFaithfulnessAnalyzerV0" in diagnosis.evidence

def test_complete_pass_remains_clean(basic_run):
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockPassAnalyzer("NCVPipelineVerifier")
    ]
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.CLEAN

def test_citation_analyzer_critical_when_citations_present(basic_run):
    run_with_citations = RAGRun(
        run_id="c-run",
        query="Q",
        retrieved_chunks=[RetrievedChunk(chunk_id="c1", text="T", source_doc_id="d1", score=1.0)],
        final_answer="A [1].",
        cited_doc_ids=["doc1"]
    )
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockPassAnalyzer("NCVPipelineVerifier"),
        MockSkipAnalyzer("CitationFaithfulnessAnalyzerV0")
    ]
    diagnosis = engine.diagnose(run_with_citations)
    assert diagnosis.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS
    assert "CitationFaithfulnessAnalyzerV0" in diagnosis.evidence

def test_citation_analyzer_not_critical_when_no_citations(basic_run):
    engine = DiagnosisEngine(config={"mode": "external-enhanced", "enabled_external_providers": []})
    engine.analyzers = [
        MockPassAnalyzer("ClaimGroundingAnalyzer"),
        MockPassAnalyzer("RetrievalDiagnosisAnalyzerV0"),
        MockPassAnalyzer("NCVPipelineVerifier"),
        MockSkipAnalyzer("CitationFaithfulnessAnalyzerV0")
    ]
    # basic_run has no cited_doc_ids
    diagnosis = engine.diagnose(basic_run)
    assert diagnosis.primary_failure == FailureType.CLEAN
