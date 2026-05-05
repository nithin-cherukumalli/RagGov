"""Tests for DeepEvalRetrievalSignalProvider (DeepEval is an optional dependency).

All tests run without a real DeepEval installation. Remote API calls are never made.
Tests use metric_runner mocks or monkeypatched importlib to simulate
presence/absence of the deepeval package.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from raggov.evaluators.retrieval.deepeval_adapter import DeepEvalRetrievalSignalProvider
from raggov.models.chunk import RetrievedChunk
from raggov.models.retrieval_diagnosis import RetrievalFailureType
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(*, include_reference: bool = False) -> RAGRun:
    metadata: dict = {}
    if include_reference:
        metadata["reference_answer"] = "Refunds are available for thirty days."
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
        metadata=metadata,
    )


def _provider_with_mock(metrics: dict, **config_overrides) -> DeepEvalRetrievalSignalProvider:
    """Return a provider whose metric_runner returns the given metrics dict."""
    return DeepEvalRetrievalSignalProvider(
        {"metric_runner": lambda _run: metrics, **config_overrides}
    )


# ---------------------------------------------------------------------------
# 1. Missing dependency
# ---------------------------------------------------------------------------

def test_missing_dependency_returns_missing_dependency(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "deepeval":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    result = DeepEvalRetrievalSignalProvider().evaluate(_run())

    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "deepeval" in result.error


# ---------------------------------------------------------------------------
# 2–4. Mocked metric output → ExternalSignalRecord
# ---------------------------------------------------------------------------

def test_mocked_contextual_precision_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.42}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["deepeval_contextual_precision"]
    assert sig.provider == ExternalEvaluatorProvider.deepeval
    assert sig.signal_type == ExternalSignalType.retrieval_contextual_precision
    assert sig.value == 0.42


def test_mocked_context_recall_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.31}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["deepeval_context_recall"]
    assert sig.signal_type == ExternalSignalType.retrieval_context_recall
    assert sig.value == 0.31


def test_mocked_faithfulness_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"faithfulness": 0.28}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["deepeval_faithfulness"]
    assert sig.signal_type == ExternalSignalType.faithfulness
    assert sig.value == 0.28


def test_mocked_contextual_relevancy_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_relevancy": 0.5}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["deepeval_contextual_relevancy"]
    assert sig.signal_type == ExternalSignalType.retrieval_contextual_relevancy
    assert sig.value == 0.5


def test_mocked_metric_output_maps_to_external_signal_records(monkeypatch) -> None:
    """All metrics together."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock(
        {"contextual_precision": 0.42, "context_recall": 0.31, "faithfulness": 0.28, "contextual_relevancy": 0.5}
    ).evaluate(_run())

    assert result.succeeded is True
    assert result.raw_payload == {
        "contextual_precision": 0.42,
        "context_recall": 0.31,
        "faithfulness": 0.28,
        "contextual_relevancy": 0.5,
    }
    by_metric = {s.metric_name: s for s in result.signals}
    assert by_metric["deepeval_contextual_precision"].provider == ExternalEvaluatorProvider.deepeval
    assert by_metric["deepeval_contextual_precision"].signal_type == ExternalSignalType.retrieval_contextual_precision
    assert by_metric["deepeval_context_recall"].signal_type == ExternalSignalType.retrieval_context_recall
    assert by_metric["deepeval_faithfulness"].signal_type == ExternalSignalType.faithfulness
    assert by_metric["deepeval_contextual_relevancy"].signal_type == ExternalSignalType.retrieval_contextual_relevancy


# ---------------------------------------------------------------------------
# 5. Missing reference input is visible
# ---------------------------------------------------------------------------

def test_missing_reference_input_visible_for_context_recall(monkeypatch) -> None:
    """context_recall requires a reference answer. If absent, emit visible signal."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    # Run without reference_answer in metadata
    result = _provider_with_mock({"context_recall": 0.5}).evaluate(_run(include_reference=False))

    assert result.succeeded is True
    # There must be a missing_reference_input signal
    missing_ref_signals = [
        s for s in result.signals if s.label == "missing_reference_input"
    ]
    assert missing_ref_signals, "Expected at least one missing_reference_input signal"
    assert missing_ref_signals[0].signal_type == ExternalSignalType.custom
    assert "reference" in missing_ref_signals[0].explanation.lower()


def test_reference_present_no_missing_reference_signal(monkeypatch) -> None:
    """When reference_answer is present, no missing_reference_input signal is emitted."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.5}).evaluate(_run(include_reference=True))

    assert result.succeeded is True
    missing_ref_signals = [
        s for s in result.signals if s.label == "missing_reference_input"
    ]
    assert not missing_ref_signals


# ---------------------------------------------------------------------------
# 6. Raw payload is preserved
# ---------------------------------------------------------------------------

def test_raw_payload_preserved(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.1}).evaluate(_run())

    assert result.raw_payload == {"context_recall": 0.1}
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_context_recall"]
    assert metric_signals, "Expected deepeval_context_recall signal"
    assert metric_signals[0].raw_payload == {"metric": "context_recall", "value": 0.1}


# ---------------------------------------------------------------------------
# 7. calibration_status = uncalibrated_locally
# ---------------------------------------------------------------------------

def test_calibration_status_uncalibrated_locally(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.5}).evaluate(_run())

    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals
    assert metric_signals[0].calibration_status == "uncalibrated_locally"


# ---------------------------------------------------------------------------
# 8. recommended_for_gating = False
# ---------------------------------------------------------------------------

def test_recommended_for_gating_false(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.1}).evaluate(_run())

    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_context_recall"]
    assert metric_signals
    assert metric_signals[0].recommended_for_gating is False


# ---------------------------------------------------------------------------
# Derived label tests
# ---------------------------------------------------------------------------

def test_derived_label_high(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.9}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals[0].label == "high"


def test_derived_label_medium(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.55}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals[0].label == "medium"


def test_derived_label_low(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.2}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals[0].label == "low"


def test_derived_label_unavailable_for_non_numeric(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": "n/a"}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals[0].label == "unavailable"


def test_derived_label_custom_thresholds(monkeypatch) -> None:
    """Thresholds are configurable; verify label changes with different config."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    # With very high threshold: 0.55 should be "low"
    result = _provider_with_mock(
        {"contextual_precision": 0.55},
        deepeval_label_high_threshold=0.90,
        deepeval_label_low_threshold=0.60,
    ).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert metric_signals[0].label == "low"


def test_limitations_mention_uncalibrated_thresholds(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"contextual_precision": 0.5}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "deepeval_contextual_precision"]
    assert any("uncalibrated" in lim for lim in metric_signals[0].limitations)


# ---------------------------------------------------------------------------
# RetrievalDiagnosis integration
# ---------------------------------------------------------------------------

def test_retrieval_diagnosis_can_include_deepeval_evidence(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock({"context_recall": 0.1}).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    assert any(
        signal.source_report == "ExternalEvaluationResult:deepeval"
        and signal.signal_name == "deepeval_context_recall"
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )


def test_no_final_diagnosis_directly_determined_by_deepeval_alone(monkeypatch) -> None:
    """DeepEval signals annotate evidence but do not alone force a retrieval failure type."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock(
        {"contextual_precision": 0.0, "context_recall": 0.0, "contextual_relevancy": 0.0}
    ).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    assert result.retrieval_diagnosis_report.primary_failure_type not in {
        RetrievalFailureType.RETRIEVAL_NOISE,
        RetrievalFailureType.RETRIEVAL_MISS,
    }


def test_low_contextual_relevancy_annotates_evidence_but_does_not_force_noise(monkeypatch) -> None:
    """Low contextual_relevancy strengthens noise evidence but does not alone classify failure."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock({"contextual_relevancy": 0.05}).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    assert any(
        "deepeval_contextual_relevancy" in signal.signal_name
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )
    assert result.retrieval_diagnosis_report.primary_failure_type != RetrievalFailureType.RETRIEVAL_NOISE


def test_low_context_recall_annotates_evidence_but_does_not_force_miss(monkeypatch) -> None:
    """Low context_recall strengthens miss evidence but does not alone classify failure."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock({"context_recall": 0.05}).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    assert any(
        "deepeval_context_recall" in signal.signal_name
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )
    assert result.retrieval_diagnosis_report.primary_failure_type != RetrievalFailureType.RETRIEVAL_MISS


# ---------------------------------------------------------------------------
# Engine / mode integration
# ---------------------------------------------------------------------------

def test_native_mode_does_not_call_deepeval() -> None:
    """In native mode, deepeval must not appear in external_signals_used."""
    from raggov.engine import DiagnosisEngine

    engine = DiagnosisEngine({"mode": "native"})
    engine.analyzers = []

    rag_run = _run()
    diagnosis = engine.diagnose(rag_run)

    assert "deepeval" not in diagnosis.external_signals_used
    assert "deepeval" not in diagnosis.missing_external_providers


def test_external_enhanced_deepeval_missing_dep_sets_degraded(monkeypatch) -> None:
    """In external-enhanced mode with DeepEval missing, degraded_external_mode=True."""
    from raggov.engine import DiagnosisEngine

    def _deepeval_unavailable(self):
        return False

    monkeypatch.setattr(DeepEvalRetrievalSignalProvider, "is_available", _deepeval_unavailable)

    # Disable RAGAS so it doesn't mask the deepeval check
    def _ragas_unavailable(self):
        return False
    from raggov.evaluators.retrieval.ragas_adapter import RagasRetrievalSignalProvider
    monkeypatch.setattr(RagasRetrievalSignalProvider, "is_available", _ragas_unavailable)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    rag_run = _run()
    rag_run.retrieved_chunks = []

    diagnosis = engine.diagnose(rag_run)

    assert diagnosis.degraded_external_mode is True
    assert "deepeval" in diagnosis.missing_external_providers
    assert "native_retrieval_signals_only" in diagnosis.fallback_heuristics_used


def test_external_enhanced_deepeval_missing_dep_via_mock(monkeypatch) -> None:
    """Engine records deepeval in missing_external_providers when dep is absent."""
    from raggov.engine import DiagnosisEngine

    def _deepeval_unavailable(self):
        return False

    monkeypatch.setattr(DeepEvalRetrievalSignalProvider, "is_available", _deepeval_unavailable)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    rag_run = _run()
    diagnosis = engine.diagnose(rag_run)

    assert "deepeval" in diagnosis.missing_external_providers


def test_ncv_sees_deepeval_influenced_retrieval_diagnosis(monkeypatch) -> None:
    """NCV consumes DeepEval effects via RetrievalDiagnosisReport without calling DeepEval directly."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())

    external_result = _provider_with_mock(
        {"contextual_precision": 0.1, "context_recall": 0.1}
    ).evaluate(_run())

    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    rag_run.retrieval_diagnosis_report = diag_result.retrieval_diagnosis_report

    assert rag_run.retrieval_diagnosis_report is not None
    deepeval_signals = [
        s for s in rag_run.retrieval_diagnosis_report.evidence_signals
        if "deepeval" in s.source_report
    ]
    assert deepeval_signals, "NCV-bound retrieval diagnosis report should contain DeepEval signals"
