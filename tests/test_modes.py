import pytest
from unittest.mock import MagicMock
from raggov.config import EngineConfig, DiagnosisMode
from raggov.engine import DiagnosisEngine


def test_diagnosis_engine_default_mode():
    engine = DiagnosisEngine({})
    assert engine.config.get("mode") == "external-enhanced"
    assert engine.config.get("enabled_external_providers") == [
        "ragas",
        "deepeval",
        "refchecker_claim",
        "refchecker_citation",
        "ragchecker",
    ]
    assert engine.config.get("retrieval_relevance_provider") == "native"
    assert engine.config.get("enable_a2p") is False


def test_diagnosis_engine_native_mode_overrides_external():
    config: EngineConfig = {
        "mode": "native",
        "has_llm": True,
        "enable_triplet_extraction": True,
        "enable_a2p": True,
    }
    engine = DiagnosisEngine(config)
    assert engine.config.get("claim_verifier") == "heuristic"
    assert engine.config.get("enable_a2p") is False
    assert engine.config.get("enabled_external_providers") == []


def test_diagnosis_engine_calibrated_mode_raises():
    with pytest.raises(NotImplementedError, match="Calibrated mode is not yet available natively."):
        DiagnosisEngine({"mode": "calibrated"})


def test_diagnosis_engine_missing_external_providers():
    config: EngineConfig = {
        "mode": "external-enhanced",
        "enabled_external_providers": ["structured_llm_claim", "a2p"],
        "llm_client": None,  # Missing LLM
        "enable_triplet_extraction": True,
    }
    engine = DiagnosisEngine(config)
    
    run = MagicMock()
    run.run_id = "test-run"
    run.model_dump.return_value = {}
    
    # We must patch analyzers so it doesn't crash on empty RAGRun Mock
    engine.analyzers = []
    
    diagnosis = engine.diagnose(run)
    assert diagnosis.diagnosis_mode == "external-enhanced"
    assert "structured_llm_claim" in diagnosis.missing_external_providers
    assert "a2p" in diagnosis.missing_external_providers
    assert "heuristic_claim_verifier" in diagnosis.fallback_heuristics_used
    assert "legacy_failure_level_heuristic" in diagnosis.fallback_heuristics_used
