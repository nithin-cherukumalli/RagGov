import pytest
import json
from pathlib import Path
from raggov.engine import DiagnosisEngine
from raggov.models.run import RAGRun
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureType, ClaimResult
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.calibration import ARESCalibrator, CalibrationSample

def _test_run(run_id="test"):
    return RAGRun(
        run_id=run_id,
        query="What is RagGov?",
        final_answer="RagGov is a diagnostic tool.",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk1",
                text="RagGov is a diagnostic tool.",
                source_doc_id="doc1",
                score=0.9
            )
        ],
        metadata={"claim_results": [{"claim_text": "RagGov is a tool", "label": "entailed", "supporting_chunk_ids": ["chunk1"]}]}
    )

def test_semantic_entropy_labeling():
    """Verify that semantic entropy paths are labeled correctly in evidence."""
    run = _test_run("test_entropy")
    
    analyzer = SemanticEntropyAnalyzer()
    # Deterministic path (default)
    result = analyzer.analyze(run)
    assert any("claim_label_entropy_proxy_v0" in e for e in result.evidence)
    assert any("Note: Using deterministic label entropy proxy" in e for e in result.evidence)

def test_score_separation():
    """Verify that heuristic, diagnostic, and calibrated scores are separated."""
    run = _test_run("test_scores")
    
    engine = DiagnosisEngine()
    diagnosis = engine.diagnose(run)
    
    # Should have diagnostic_score (from semantic entropy) and heuristic_score (from others)
    assert diagnosis.diagnostic_score is not None
    assert diagnosis.heuristic_score is not None
    assert diagnosis.calibrated_confidence is None
    assert diagnosis.calibration_status == "uncalibrated"
    assert "Confidence Signal (uncalibrated)" in diagnosis.confidence_label

def test_calibration_gating():
    """Verify that calibrated_confidence is only populated with a valid artifact."""
    # Create 30 samples for minimum calibration
    samples = []
    for i in range(30):
        samples.append({
            "run_id": f"run_{i}",
            "automated_faithfulness": 0.8,
            "automated_retrieval_precision": 0.7,
            "automated_answer_correctness": 0.9,
            "gold_faithfulness": 0.85,
            "gold_retrieval_precision": 0.75,
            "gold_answer_correctness": 0.95
        })
    
    artifact = {"samples": samples, "confidence_level": 0.95}
    
    run = _test_run("test_calibrated")
    
    # Test with 30 samples (PROVISIONAL)
    engine = DiagnosisEngine(config={"calibrator": artifact})
    diagnosis = engine.diagnose(run)
    
    assert diagnosis.calibrated_confidence is not None
    assert diagnosis.calibration_status == "provisional"
    assert "Provisional Confidence" in diagnosis.confidence_label
    
    # Test with 150 samples (CALIBRATED)
    samples_150 = []
    for i in range(150):
        samples_150.append({
            "run_id": f"run_{i}",
            "automated_faithfulness": 0.8,
            "automated_retrieval_precision": 0.7,
            "automated_answer_correctness": 0.9,
            "gold_faithfulness": 0.85,
            "gold_retrieval_precision": 0.75,
            "gold_answer_correctness": 0.95
        })
    
    artifact_150 = {"samples": samples_150, "confidence_level": 0.95}
    engine_150 = DiagnosisEngine(config={"calibrator": artifact_150})
    diagnosis_150 = engine_150.diagnose(run)
    
    assert diagnosis_150.calibration_status == "calibrated"
    assert "Calibrated Confidence" in diagnosis_150.confidence_label

def test_calibration_min_samples():
    """Verify that calibration fails with fewer than 30 samples."""
    calibrator = ARESCalibrator()
    for i in range(10):
        calibrator.add_sample(CalibrationSample(
            run_id=f"run_{i}",
            automated_faithfulness=0.8,
            automated_retrieval_precision=0.7,
            automated_answer_correctness=0.9,
            gold_faithfulness=0.8,
            gold_retrieval_precision=0.7,
            gold_answer_correctness=0.9
        ))
    
    with pytest.raises(ValueError, match="Calibration requires at least 30 samples"):
        calibrator.calibrate_with_status()
