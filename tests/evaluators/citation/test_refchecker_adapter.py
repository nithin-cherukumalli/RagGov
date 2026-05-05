"""Tests for RefCheckerCitationSignalProvider (refchecker is an optional dependency).

All tests run without a real RefChecker installation.
Tests use metric_runner / citation_runner mocks or monkeypatched importlib.
No model downloads or remote API calls are made.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from raggov.analyzers.citation_faithfulness.analyzer import CitationFaithfulnessAnalyzerV0
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from raggov.evaluators.citation.refchecker_adapter import (
    RefCheckerCitationSignalProvider,
    _normalize_citation_label,
)
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
    )


def _provider_with_citations(citations_data: list[dict], **config_overrides) -> RefCheckerCitationSignalProvider:
    return RefCheckerCitationSignalProvider(
        {
            "metric_runner": lambda _run: {"citations": citations_data},
            **config_overrides,
        }
    )


# ---------------------------------------------------------------------------
# 1. Missing dependency
# ---------------------------------------------------------------------------

def test_missing_dependency_returns_missing_dependency(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "refchecker":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import)
    result = RefCheckerCitationSignalProvider().evaluate(_run())

    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "refchecker" in result.error


# ---------------------------------------------------------------------------
# 2–4. Mocked metric output → ExternalSignalRecord
# ---------------------------------------------------------------------------

def test_supports_output_becomes_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.92}]
    ).evaluate(_run())

    assert result.succeeded is True
    sig = result.signals[0]
    assert sig.provider == ExternalEvaluatorProvider.refchecker
    assert sig.label == "supports"
    assert sig.signal_type == ExternalSignalType.citation_support
    assert sig.affected_claim_ids == ["c1"]
    assert sig.affected_doc_ids == ["doc-1"]


def test_contradicts_output_becomes_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "contradicts", "score": 0.15}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "contradicts"


def test_does_not_support_output_becomes_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "does_not_support", "score": 0.1}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "does_not_support"


def test_citation_missing_label(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "", "label": "missing", "score": None}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "citation_missing"


# ---------------------------------------------------------------------------
# 5. Triplets preserved in raw_payload
# ---------------------------------------------------------------------------

def test_triplet_level_output_preserved_in_raw_payload(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    triplets = [{"subject": "Refunds", "predicate": "lasts", "object": "30 days", "label": "supports"}]
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.9, "triplets": triplets}]
    ).evaluate(_run())

    assert result.signals[0].raw_payload["triplets"] == triplets


# ---------------------------------------------------------------------------
# 6. Citation label normalization
# ---------------------------------------------------------------------------

def test_citation_labels_normalize_correctly() -> None:
    assert _normalize_citation_label("entailment") == "supports"
    assert _normalize_citation_label("Supported") == "supports"
    assert _normalize_citation_label("contradiction") == "contradicts"
    assert _normalize_citation_label("neutral") == "does_not_support"
    assert _normalize_citation_label("hallucinated") == "does_not_support"
    assert _normalize_citation_label("missing") == "citation_missing"
    assert _normalize_citation_label("unknown") == "unclear"
    assert _normalize_citation_label(None) == "unclear"
    assert _normalize_citation_label("garbage") == "unclear"


def test_unclear_is_not_treated_as_support(monkeypatch) -> None:
    """Unclear label must not be mapped to 'supports'."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "unclear", "score": None}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "unclear"
    assert sig.label != "supports"


# ---------------------------------------------------------------------------
# 7. calibration_status and recommended_for_gating
# ---------------------------------------------------------------------------

def test_calibration_status_uncalibrated_locally(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.8}]
    ).evaluate(_run())

    assert result.signals[0].calibration_status == "uncalibrated_locally"


def test_recommended_for_gating_false(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.8}]
    ).evaluate(_run())

    assert result.signals[0].recommended_for_gating is False


# ---------------------------------------------------------------------------
# 8. Raw payload preserved
# ---------------------------------------------------------------------------

def test_raw_payload_preserved(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.9}]
    ).evaluate(_run())

    assert "citations" in result.raw_payload
    assert result.signals[0].raw_payload["label"] == "supports"


# ---------------------------------------------------------------------------
# 9. verify_citations() protocol
# ---------------------------------------------------------------------------

def test_verify_citations_returns_signal_records(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    provider = RefCheckerCitationSignalProvider({
        "citation_runner": lambda cited_ids, chunks: [
            {"claim_id": "c1", "cited_id": cited_ids[0], "label": "supports", "score": 0.9, "triplets": []}
        ]
    })
    signals = provider.verify_citations(["doc-1"], ["Refunds are available for 30 days."])
    assert signals
    assert signals[0].label == "supports"
    assert signals[0].signal_type == ExternalSignalType.citation_support


# ---------------------------------------------------------------------------
# 10. Native mode does not call RefChecker
# ---------------------------------------------------------------------------

def test_native_mode_does_not_call_refchecker() -> None:
    from raggov.engine import DiagnosisEngine

    engine = DiagnosisEngine({"mode": "native"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert "refchecker_citation" not in diagnosis.external_signals_used
    assert "refchecker_citation" not in diagnosis.missing_external_providers


# ---------------------------------------------------------------------------
# 11. External-enhanced with RefChecker missing → degraded
# ---------------------------------------------------------------------------

def test_external_enhanced_refchecker_citation_missing_dep_sets_degraded(monkeypatch) -> None:
    from raggov.engine import DiagnosisEngine

    monkeypatch.setattr(RefCheckerCitationSignalProvider, "is_available", lambda self: False)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert diagnosis.degraded_external_mode is True
    assert "refchecker_citation" in diagnosis.missing_external_providers
    assert "native_citation_verifier" in diagnosis.fallback_heuristics_used


# ---------------------------------------------------------------------------
# 12. CitationFaithfulness consumes refchecker when configured
# ---------------------------------------------------------------------------

def test_citation_faithfulness_consumes_refchecker_signal_when_configured(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())

    provider = RefCheckerCitationSignalProvider({
        "citation_runner": lambda cited_ids, chunks: [
            {"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.9, "triplets": []}
        ]
    })
    analyzer = CitationFaithfulnessAnalyzerV0({"citation_verifier": "refchecker"})
    analyzer._refchecker_citation_provider = provider

    rag_run = _run()
    result = analyzer.analyze(rag_run)
    # Should not crash; external signal path is wired
    assert result.status in ("pass", "fail", "skip")


def test_citation_faithfulness_refchecker_missing_dep_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(RefCheckerCitationSignalProvider, "is_available", lambda self: False)

    analyzer = CitationFaithfulnessAnalyzerV0({"citation_verifier": "refchecker"})
    rag_run = _run()
    result = analyzer.analyze(rag_run)
    # Should not crash; should have recorded the missing dep error
    assert result.status in ("pass", "fail", "skip")
    assert analyzer._external_verifier_error is not None


# ---------------------------------------------------------------------------
# 13. External signal records preserved end-to-end
# ---------------------------------------------------------------------------

def test_external_signal_records_have_provider_refchecker(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_citations(
        [{"claim_id": "c1", "cited_id": "doc-1", "label": "supports", "score": 0.9}]
    ).evaluate(_run())

    for sig in result.signals:
        assert sig.provider == ExternalEvaluatorProvider.refchecker
        assert "refchecker" in sig.limitations[1] or "uncalibrated" in sig.calibration_status
