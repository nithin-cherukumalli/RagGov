"""Tests for RagasRetrievalSignalProvider (RAGAS is an optional dependency).

All tests run without a real RAGAS installation. Remote API calls are never made.
Tests use metric_runner mocks or monkeypatched importlib to simulate
presence/absence of the ragas package.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from raggov.evaluators.retrieval.ragas_adapter import RagasRetrievalSignalProvider
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


def _provider_with_mock(metrics: dict, **config_overrides) -> RagasRetrievalSignalProvider:
    """Return a provider whose metric_runner returns the given metrics dict."""
    return RagasRetrievalSignalProvider(
        {"metric_runner": lambda _run: metrics, **config_overrides}
    )


# ---------------------------------------------------------------------------
# 1. Missing dependency
# ---------------------------------------------------------------------------

def test_missing_dependency_returns_missing_dependency(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "ragas":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    result = RagasRetrievalSignalProvider().evaluate(_run())

    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "ragas" in result.error


# ---------------------------------------------------------------------------
# 2–4. Mocked metric output → ExternalSignalRecord
# ---------------------------------------------------------------------------

def test_mocked_context_precision_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.42}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["ragas_context_precision"]
    assert sig.provider == ExternalEvaluatorProvider.ragas
    assert sig.signal_type == ExternalSignalType.retrieval_context_precision
    assert sig.value == 0.42


def test_mocked_context_recall_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.31}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["ragas_context_recall"]
    assert sig.signal_type == ExternalSignalType.retrieval_context_recall
    assert sig.value == 0.31


def test_mocked_faithfulness_maps_to_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"faithfulness": 0.28}).evaluate(_run())

    assert result.succeeded is True
    by_metric = {s.metric_name: s for s in result.signals}
    sig = by_metric["ragas_faithfulness"]
    assert sig.signal_type == ExternalSignalType.faithfulness
    assert sig.value == 0.28


def test_mocked_metric_output_maps_to_external_signal_records(monkeypatch) -> None:
    """All three metrics together."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock(
        {"context_precision": 0.42, "context_recall": 0.31, "faithfulness": 0.28}
    ).evaluate(_run())

    assert result.succeeded is True
    assert result.raw_payload == {
        "context_precision": 0.42,
        "context_recall": 0.31,
        "faithfulness": 0.28,
    }
    by_metric = {s.metric_name: s for s in result.signals}
    assert by_metric["ragas_context_precision"].provider == ExternalEvaluatorProvider.ragas
    assert by_metric["ragas_context_precision"].signal_type == ExternalSignalType.retrieval_context_precision
    assert by_metric["ragas_context_recall"].signal_type == ExternalSignalType.retrieval_context_recall
    assert by_metric["ragas_faithfulness"].signal_type == ExternalSignalType.faithfulness


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
    # Find the actual metric signal (not the missing_reference_input signal)
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_recall"]
    assert metric_signals, "Expected ragas_context_recall signal"
    assert metric_signals[0].raw_payload == {"metric": "context_recall", "value": 0.1}


# ---------------------------------------------------------------------------
# 7. calibration_status = uncalibrated_locally
# ---------------------------------------------------------------------------

def test_calibration_status_uncalibrated_locally(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.5}).evaluate(_run())

    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals
    assert metric_signals[0].calibration_status == "uncalibrated_locally"


# ---------------------------------------------------------------------------
# 8. recommended_for_gating = False
# ---------------------------------------------------------------------------

def test_recommended_for_gating_false(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_recall": 0.1}).evaluate(_run())

    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_recall"]
    assert metric_signals
    assert metric_signals[0].recommended_for_gating is False


# ---------------------------------------------------------------------------
# Derived label tests
# ---------------------------------------------------------------------------

def test_derived_label_high(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.9}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals[0].label == "high"


def test_derived_label_medium(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.55}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals[0].label == "medium"


def test_derived_label_low(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.2}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals[0].label == "low"


def test_derived_label_unavailable_for_non_numeric(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": "n/a"}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals[0].label == "unavailable"


def test_derived_label_custom_thresholds(monkeypatch) -> None:
    """Thresholds are configurable; verify label changes with different config."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    # With very high threshold: 0.55 should be "low"
    result = _provider_with_mock(
        {"context_precision": 0.55},
        ragas_label_high_threshold=0.90,
        ragas_label_low_threshold=0.60,
    ).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert metric_signals[0].label == "low"


def test_limitations_mention_uncalibrated_thresholds(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_mock({"context_precision": 0.5}).evaluate(_run())
    metric_signals = [s for s in result.signals if s.metric_name == "ragas_context_precision"]
    assert any("uncalibrated" in lim for lim in metric_signals[0].limitations)


# ---------------------------------------------------------------------------
# RetrievalDiagnosis integration
# ---------------------------------------------------------------------------

def test_retrieval_diagnosis_can_include_ragas_evidence(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock({"context_recall": 0.1}).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    assert any(
        signal.source_report == "ExternalEvaluationResult:ragas"
        and signal.signal_name == "ragas_context_recall"
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )


def test_no_final_diagnosis_directly_determined_by_ragas_alone(monkeypatch) -> None:
    """RAGAS signals annotate evidence but do not alone force a retrieval failure type."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock(
        {"context_precision": 0.0, "context_recall": 0.0}
    ).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    # RAGAS alone with otherwise-sufficient native context must not trigger noise/miss.
    assert result.retrieval_diagnosis_report.primary_failure_type not in {
        RetrievalFailureType.RETRIEVAL_NOISE,
        RetrievalFailureType.RETRIEVAL_MISS,
    }


def test_low_context_precision_annotates_evidence_but_does_not_force_noise(monkeypatch) -> None:
    """Low context_precision strengthens noise evidence but does not alone classify failure."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    external_result = _provider_with_mock({"context_precision": 0.05}).evaluate(_run())
    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)

    assert result.retrieval_diagnosis_report is not None
    # The signal is surfaced in evidence_signals...
    assert any(
        "ragas_context_precision" in signal.signal_name
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )
    # ...but without other corroborating evidence, primary_failure_type is not RETRIEVAL_NOISE.
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
        "ragas_context_recall" in signal.signal_name
        for signal in result.retrieval_diagnosis_report.evidence_signals
    )
    assert result.retrieval_diagnosis_report.primary_failure_type != RetrievalFailureType.RETRIEVAL_MISS


# ---------------------------------------------------------------------------
# Engine / mode integration
# ---------------------------------------------------------------------------

def test_native_mode_does_not_call_ragas() -> None:
    """In native mode, ragas must not appear in external_signals_used."""
    from raggov.engine import DiagnosisEngine

    engine = DiagnosisEngine({"mode": "native"})
    engine.analyzers = []  # skip all analyzers for speed

    rag_run = _run()
    diagnosis = engine.diagnose(rag_run)

    assert "ragas" not in diagnosis.external_signals_used
    assert "ragas" not in diagnosis.missing_external_providers


def test_external_enhanced_ragas_missing_dep_sets_degraded(monkeypatch) -> None:
    """In external-enhanced mode with RAGAS missing, degraded_external_mode=True."""
    from raggov.engine import DiagnosisEngine

    # Make RAGAS appear unavailable inside the registry adapter
    original_is_available = RagasRetrievalSignalProvider.is_available

    def _ragas_unavailable(self):
        return False

    monkeypatch.setattr(RagasRetrievalSignalProvider, "is_available", _ragas_unavailable)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    rag_run = _run()
    # Need to satisfy mock for retrieval evidence
    rag_run.retrieved_chunks = []

    diagnosis = engine.diagnose(rag_run)

    assert diagnosis.degraded_external_mode is True
    assert "ragas" in diagnosis.missing_external_providers
    assert "native_retrieval_signals_only" in diagnosis.fallback_heuristics_used


def test_external_enhanced_ragas_missing_dep_via_mock(monkeypatch) -> None:
    """Engine records ragas in missing_external_providers when dep is absent."""
    from raggov.engine import DiagnosisEngine

    original_is_available = RagasRetrievalSignalProvider.is_available

    def _ragas_unavailable(self):
        return False

    monkeypatch.setattr(RagasRetrievalSignalProvider, "is_available", _ragas_unavailable)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    rag_run = _run()
    diagnosis = engine.diagnose(rag_run)

    assert "ragas" in diagnosis.missing_external_providers


def test_ncv_sees_ragas_influenced_retrieval_diagnosis(monkeypatch) -> None:
    """NCV consumes RAGAS effects via RetrievalDiagnosisReport without calling RAGAS directly."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())

    external_result = _provider_with_mock(
        {"context_precision": 0.1, "context_recall": 0.1}
    ).evaluate(_run())

    rag_run = _run()
    rag_run.metadata["external_evaluation_results"] = [external_result]

    # Run RetrievalDiagnosisAnalyzer to compute a report (NCV consumes this downstream).
    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    rag_run.retrieval_diagnosis_report = diag_result.retrieval_diagnosis_report

    # The report exists and contains RAGAS evidence signals.
    assert rag_run.retrieval_diagnosis_report is not None
    ragas_signals = [
        s for s in rag_run.retrieval_diagnosis_report.evidence_signals
        if "ragas" in s.source_report
    ]
    assert ragas_signals, "NCV-bound retrieval diagnosis report should contain RAGAS signals"
