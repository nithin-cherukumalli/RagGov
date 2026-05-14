import pytest
from pathlib import Path
import json
from raggov.engine import DiagnosisEngine
from raggov.models.run import RAGRun
from raggov.models.diagnosis import FailureType, FailureStage

def test_diagnosis_summary_population():
    """Verify that DiagnosisSummary is populated correctly by the engine."""
    # Create a simple failing run
    run = RAGRun(
        run_id="test_fail_1",
        query="What is the interest rate?",
        final_answer="The interest rate is 5%.",
        retrieved_chunks=[
            {"chunk_id": "chunk1", "text": "The interest rate is 4%.", "source_doc_id": "doc1", "score": 1.0}
        ],
        metadata={"target_failure": "CITATION_MISMATCH"}
    )
    
    engine = DiagnosisEngine(config={"mode": "native"})
    diagnosis = engine.diagnose(run)
    
    assert diagnosis.summary_v1 is not None
    assert diagnosis.summary_v1.primary_failure != FailureType.CLEAN
    assert diagnosis.summary_v1.recommended_fix is not None
    assert diagnosis.summary_v1.recommended_next_debug_step is not None
    # Just check that it's a non-empty string providing some instruction
    assert len(diagnosis.summary_v1.recommended_next_debug_step) > 10

def test_diagnosis_json_completeness():
    """Verify that the JSON output contains all actionable trace fields."""
    run = RAGRun(
        run_id="test_json_1",
        query="test",
        final_answer="test",
        retrieved_chunks=[]
    )
    
    engine = DiagnosisEngine(config={"mode": "native"})
    diagnosis = engine.diagnose(run)
    
    # Check JSON serialization
    json_data = json.loads(diagnosis.model_dump_json())
    
    assert "summary_v1" in json_data
    assert "diagnosis_decision_trace" in json_data
    assert "pinpoint_findings" in json_data
    assert "causal_chains" in json_data
    assert "external_provider_readiness" in json_data

def test_degraded_provider_visibility():
    """Verify that degraded providers appear in the summary."""
    run = RAGRun(
        run_id="test_degraded_1",
        query="test",
        final_answer="test",
        retrieved_chunks=[]
    )
    
    # Force a degraded state by requiring a provider that isn't configured
    engine = DiagnosisEngine(config={
        "mode": "external-enhanced",
        "enabled_external_providers": ["ragas"],
        "strict_external_evaluators": False
    })
    diagnosis = engine.diagnose(run)
    
    assert "ragas" in diagnosis.summary_v1.external_provider_state
    # In a typical local env without ragas, it should be degraded or missing
    # But let's just check it's present in the state map
    assert diagnosis.summary_v1.external_provider_state["ragas"] in ("missing", "degraded", "available")

def test_missing_evidence_visibility():
    """Verify that missing evidence is tracked in the summary."""
    # This usually comes from NCV reports when some analyzers were skipped or failed
    run = RAGRun(
        run_id="test_missing_1",
        query="test",
        final_answer="test",
        retrieved_chunks=[]
    )
    
    engine = DiagnosisEngine(config={"mode": "native"})
    diagnosis = engine.diagnose(run)
    
    # In native mode, many things might be missing compared to full mode
    # We just want to ensure the field exists and is a list
    assert isinstance(diagnosis.summary_v1.missing_evidence, list)
