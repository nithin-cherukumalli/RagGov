"""Tests for RAGCheckerSignalProvider (ragchecker is an optional dependency).

All tests run without a real RAGChecker installation.
Tests use metric_runner mocks or monkeypatched importlib.
No model downloads or remote API calls are made.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.evaluators.base import (
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.evaluators.retrieval.ragchecker_adapter import RAGCheckerSignalProvider
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run() -> RAGRun:
    return RAGRun(
        query="What is the refund policy?",
        final_answer="Refunds are available for thirty days.",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                source_doc_id="doc-1",
                text="Refunds are available for thirty days.",
                score=None,
            )
        ],
        metadata={
            "reference_answer": "Users can get a refund within 30 days of purchase.",
            "gold_claims": ["Refunds available for 30 days"]
        }
    )

def _provider_with_metrics(metrics: dict, **config_overrides) -> RAGCheckerSignalProvider:
    return RAGCheckerSignalProvider(
        {
            "metric_runner": lambda _run: metrics,
            **config_overrides,
        }
    )


# ---------------------------------------------------------------------------
# 1. Missing dependency
# ---------------------------------------------------------------------------

def test_missing_dependency_returns_missing_dependency(monkeypatch) -> None:
    def fake_import(name: str, package=None):
        if name == "ragchecker":
            return None
        return importlib.util.find_spec(name, package)

    monkeypatch.setattr("importlib.util.find_spec", fake_import)
    result = RAGCheckerSignalProvider().evaluate(_run())

    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "ragchecker" in result.error


# ---------------------------------------------------------------------------
# 2-6. Mocked metric outputs
# ---------------------------------------------------------------------------

def test_mocked_claim_recall_becomes_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"claim_recall": 0.8}).evaluate(_run())

    assert result.succeeded is True
    sig = result.signals[0]
    assert sig.provider == ExternalEvaluatorProvider.ragchecker
    assert sig.metric_name == "ragchecker_claim_recall"
    assert sig.signal_type == ExternalSignalType.claim_recall
    assert sig.value == 0.8
    assert sig.label == "high"


def test_mocked_context_precision_becomes_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"context_precision": 0.2}).evaluate(_run())

    sig = result.signals[0]
    assert sig.metric_name == "ragchecker_context_precision"
    assert sig.signal_type == ExternalSignalType.retrieval_context_precision
    assert sig.value == 0.2
    assert sig.label == "low"


def test_mocked_context_utilization_becomes_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"context_utilization": 0.5}).evaluate(_run())

    sig = result.signals[0]
    assert sig.metric_name == "ragchecker_context_utilization"
    assert sig.signal_type == ExternalSignalType.context_utilization
    assert sig.value == 0.5
    assert sig.label == "medium"


def test_mocked_faithfulness_becomes_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"faithfulness": 0.9}).evaluate(_run())

    sig = result.signals[0]
    assert sig.metric_name == "ragchecker_faithfulness"
    assert sig.signal_type == ExternalSignalType.faithfulness


def test_mocked_hallucination_becomes_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    # Hallucination has inverse logic: 0.1 is high quality (low hallucination)
    result = _provider_with_metrics({"hallucination": 0.1}).evaluate(_run())

    sig = result.signals[0]
    assert sig.metric_name == "ragchecker_hallucination"
    assert sig.signal_type == ExternalSignalType.hallucination
    assert sig.value == 0.1
    assert sig.label == "high"


# ---------------------------------------------------------------------------
# 7. Missing reference answers
# ---------------------------------------------------------------------------

def test_missing_reference_emits_missing_reference_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    
    run_no_ref = _run()
    run_no_ref.metadata = {}  # clear references
    
    result = _provider_with_metrics({"claim_recall": 0.8}).evaluate(run_no_ref)
    
    assert result.succeeded is True
    # One for the missing reference, one for the actual metric mapping
    assert len(result.signals) == 2
    
    missing_sig = next(s for s in result.signals if s.label == "missing_reference_input")
    assert missing_sig.signal_type == ExternalSignalType.custom
    assert "ragchecker_claim_recall_missing_reference" == missing_sig.metric_name


# ---------------------------------------------------------------------------
# 8-10. Payload, Calibration, Gating
# ---------------------------------------------------------------------------

def test_raw_payload_preserved(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"faithfulness": 0.9, "hallucination": 0.1}).evaluate(_run())

    assert result.raw_payload == {"faithfulness": 0.9, "hallucination": 0.1}
    assert result.signals[0].raw_payload["value"] == 0.9


def test_calibration_status_uncalibrated_locally(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"faithfulness": 0.9}).evaluate(_run())

    assert result.signals[0].calibration_status == "uncalibrated_locally"
    assert any("calibrated" in l for l in result.signals[0].limitations)


def test_recommended_for_gating_false(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    result = _provider_with_metrics({"faithfulness": 0.9}).evaluate(_run())

    assert result.signals[0].recommended_for_gating is False


# ---------------------------------------------------------------------------
# 11-12. Engine Integration
# ---------------------------------------------------------------------------

def test_native_mode_does_not_call_ragchecker() -> None:
    from raggov.engine import DiagnosisEngine

    engine = DiagnosisEngine({"mode": "native"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert "ragchecker" not in diagnosis.external_signals_used
    assert "ragchecker" not in diagnosis.missing_external_providers


def test_external_enhanced_ragchecker_missing_dep_sets_degraded(monkeypatch) -> None:
    from raggov.engine import DiagnosisEngine

    monkeypatch.setattr(RAGCheckerSignalProvider, "is_available", lambda self: False)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert diagnosis.degraded_external_mode is True
    assert "ragchecker" in diagnosis.missing_external_providers
    assert "native_retrieval_signals_only" in diagnosis.fallback_heuristics_used


# ---------------------------------------------------------------------------
# 13-18. Retrieval Diagnosis
# ---------------------------------------------------------------------------

def test_retrieval_diagnosis_includes_ragchecker_evidence_signals(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    
    from raggov.evaluators.base import ExternalEvaluationResult
    external_result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragchecker,
        succeeded=True,
        signals=[
            ExternalSignalRecord(
                provider=ExternalEvaluatorProvider.ragchecker,
                signal_type=ExternalSignalType.retrieval_context_precision,
                metric_name="ragchecker_context_precision",
                value=0.2,
                label="low",
            )
        ],
        raw_payload={},
    )
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    report = diag_result.retrieval_diagnosis_report
    
    assert report is not None
    assert any(sig.signal_name == "ragchecker_context_precision" for sig in report.evidence_signals)
    
    sig = next(sig for sig in report.evidence_signals if sig.signal_name == "ragchecker_context_precision")
    assert "low precision/relevancy is advisory evidence for retrieval_noise" in sig.interpretation


def test_context_utilization_interpretation(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    
    from raggov.evaluators.base import ExternalEvaluationResult
    external_result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragchecker,
        succeeded=True,
        signals=[
            ExternalSignalRecord(
                provider=ExternalEvaluatorProvider.ragchecker,
                signal_type=ExternalSignalType.context_utilization,
                metric_name="ragchecker_context_utilization",
                value=0.2,
                label="low",
            )
        ],
        raw_payload={},
    )
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    report = diag_result.retrieval_diagnosis_report
    
    sig = next(sig for sig in report.evidence_signals if sig.signal_name == "ragchecker_context_utilization")
    assert "low context utilization suggests generation may have ignored retrieved evidence" in sig.interpretation


def test_hallucination_interpretation(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    
    from raggov.evaluators.base import ExternalEvaluationResult
    external_result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.ragchecker,
        succeeded=True,
        signals=[
            ExternalSignalRecord(
                provider=ExternalEvaluatorProvider.ragchecker,
                signal_type=ExternalSignalType.hallucination,
                metric_name="ragchecker_hallucination",
                value=0.8,
                label="low", # low quality -> high hallucination
            )
        ],
        raw_payload={},
    )
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    report = diag_result.retrieval_diagnosis_report
    
    sig = next(sig for sig in report.evidence_signals if sig.signal_name == "ragchecker_hallucination")
    assert "high hallucination is advisory evidence for generation/claim issues, not retrieval alone" in sig.interpretation
