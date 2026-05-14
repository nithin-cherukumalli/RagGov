"""Tests for subtle RAG failure golden cases."""

from __future__ import annotations

import pytest
from raggov.engine import DiagnosisEngine
from raggov.models.diagnosis import FailureType
from stresslab.runners.rag_failure_runner import RAGFailureRunner
from stresslab.cases.load import load_subtle_rag_failures

@pytest.fixture
def subtle_cases():
    return load_subtle_rag_failures()

@pytest.mark.parametrize("case", load_subtle_rag_failures(), ids=lambda c: c.case_id)
def test_subtle_failure_not_clean(case):
    """Subtle failures should not return CLEAN (except explicit control cases)."""
    runner = RAGFailureRunner(mode="native", mock_native=True, suite="subtle")
    run = runner._build_run(case)
    
    # Create engine with mock analyzers
    from raggov.analyzers.base import BaseAnalyzer
    from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
    
    class _MockAnalyzer(BaseAnalyzer):
        def __init__(self, res: AnalyzerResult):
            super().__init__()
            self._res = res
        def analyze(self, r):
            return self._res

    mock_analyzers = []
    mock_analyzers.append(_MockAnalyzer(
        AnalyzerResult(
            analyzer_name="GoldenPrimaryMock",
            status="fail" if case.expected_primary_failure != FailureType.CLEAN.value else "pass",
            failure_type=FailureType(case.expected_primary_failure),
            stage=FailureStage(case.expected_root_cause_stage),
        )
    ))
    
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=mock_analyzers)
    diagnosis = engine.diagnose(run)
    
    if case.case_id != "subtle_external_disagreement_09":
        assert diagnosis.primary_failure.value != FailureType.CLEAN.value, f"Case {case.case_id} returned CLEAN"

@pytest.mark.parametrize("case", load_subtle_rag_failures(), ids=lambda c: c.case_id)
def test_subtle_failure_human_review(case):
    """Subtle failures should often recommend human review."""
    runner = RAGFailureRunner(mode="native", mock_native=True, suite="subtle")
    run = runner._build_run(case)
    
    from raggov.analyzers.base import BaseAnalyzer
    from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
    
    class _MockAnalyzer(BaseAnalyzer):
        def __init__(self, res: AnalyzerResult):
            super().__init__()
            self._res = res
        def analyze(self, r):
            return self._res

    mock_analyzers = []
    mock_analyzers.append(_MockAnalyzer(
        AnalyzerResult(
            analyzer_name="GoldenPrimaryMock",
            status="fail" if case.expected_primary_failure != FailureType.CLEAN.value else "pass",
            failure_type=FailureType(case.expected_primary_failure),
            stage=FailureStage(case.expected_root_cause_stage),
        )
    ))
    
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=mock_analyzers)
    diagnosis = engine.diagnose(run)
    
    if case.expected_human_review_required:
        assert diagnosis.human_review_required, f"Case {case.case_id} should require human review"

@pytest.mark.parametrize("case", load_subtle_rag_failures(), ids=lambda c: c.case_id)
def test_subtle_failure_external_signals(case):
    """Verify external signals are correctly incorporated."""
    if not case.expected_external_signals:
        return

    runner = RAGFailureRunner(mode="external-enhanced", mock_native=True, suite="subtle")
    run = runner._build_run(case)
    
    from raggov.analyzers.base import BaseAnalyzer
    from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
    
    class _MockAnalyzer(BaseAnalyzer):
        def __init__(self, res: AnalyzerResult):
            super().__init__()
            self._res = res
        def analyze(self, r):
            return self._res

    mock_analyzers = []
    mock_analyzers.append(_MockAnalyzer(
        AnalyzerResult(
            analyzer_name="GoldenPrimaryMock",
            status="pass" if case.expected_primary_failure == FailureType.CLEAN.value else "fail",
            failure_type=FailureType(case.expected_primary_failure),
            stage=FailureStage(case.expected_root_cause_stage),
        )
    ))
    
    providers = list(set(s["provider"] for s in case.expected_external_signals))
    engine = DiagnosisEngine(
        config={
            "mode": "external-enhanced",
            "enabled_external_providers": providers
        }, 
        analyzers=mock_analyzers
    )
    
    # Mock external registry to return the signals we expect
    from raggov.evaluators.base import ExternalEvaluationResult, ExternalSignalRecord, ExternalEvaluatorProvider
    from unittest.mock import MagicMock
    
    mock_eval_results = []
    for provider in providers:
        signals = [
            ExternalSignalRecord.model_validate(s) 
            for s in case.expected_external_signals 
            if s["provider"] == provider
        ]
        mock_eval_results.append(ExternalEvaluationResult(
            provider=ExternalEvaluatorProvider(provider),
            succeeded=True,
            signals=signals
        ))
    
    engine.external_registry.evaluate_enabled = MagicMock(return_value=mock_eval_results)
    
    diagnosis = engine.diagnose(run)
    
    assert len(diagnosis.external_diagnosis_probes) > 0, f"Case {case.case_id} should have external probes"
