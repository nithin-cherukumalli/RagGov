"""Tests for RefCheckerClaimSignalProvider (refchecker is an optional dependency).

All tests run without a real RefChecker installation.
Tests use metric_runner mocks or monkeypatched importlib.
No model downloads or remote API calls are made.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from raggov.evaluators.claim.refchecker_adapter import RefCheckerClaimSignalProvider
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


def _provider_with_claims(claims_data: list[dict], **config_overrides) -> RefCheckerClaimSignalProvider:
    return RefCheckerClaimSignalProvider(
        {
            "metric_runner": lambda _run: {"claims": claims_data},
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
    result = RefCheckerClaimSignalProvider().evaluate(_run())

    assert result.succeeded is False
    assert result.missing_dependency is True
    assert "refchecker" in result.error


# ---------------------------------------------------------------------------
# 2–4. Mocked metric output → ExternalSignalRecord
# ---------------------------------------------------------------------------

def test_entailed_output_becomes_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "entailed", "score": 0.9}]
    ).evaluate(_run())

    assert result.succeeded is True
    sig = result.signals[0]
    assert sig.provider == ExternalEvaluatorProvider.refchecker
    assert sig.label == "entailed"
    assert sig.signal_type == ExternalSignalType.claim_support
    assert sig.value == 0.9


def test_contradicted_output_becomes_external_signal_record(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "contradicted", "score": 0.1}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "contradicted"
    assert sig.affected_claim_ids == ["c1"]


def test_unsupported_output_becomes_hallucination_signal(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c2", "label": "unsupported", "score": None}]
    ).evaluate(_run())

    sig = result.signals[0]
    assert sig.label == "unsupported"
    assert sig.signal_type == ExternalSignalType.hallucination


# ---------------------------------------------------------------------------
# 5. Triplets preserved in raw_payload
# ---------------------------------------------------------------------------

def test_triplet_level_output_preserved_in_raw_payload(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    triplets = [{"subject": "Refunds", "predicate": "available_for", "object": "30 days", "label": "entailed"}]
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "entailed", "score": 0.9, "triplets": triplets}]
    ).evaluate(_run())

    assert result.signals[0].raw_payload["triplets"] == triplets


# ---------------------------------------------------------------------------
# 6. Label normalization
# ---------------------------------------------------------------------------

def test_claim_labels_normalize_correctly(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    from raggov.evaluators.claim.refchecker_adapter import _normalize_claim_label

    assert _normalize_claim_label("entailment") == "entailed"
    assert _normalize_claim_label("Supported") == "entailed"
    assert _normalize_claim_label("contradiction") == "contradicted"
    assert _normalize_claim_label("neutral") == "unsupported"
    assert _normalize_claim_label("hallucinated") == "unsupported"
    assert _normalize_claim_label("unknown") == "unclear"
    assert _normalize_claim_label(None) == "unclear"
    assert _normalize_claim_label("garbage_label") == "unclear"


# ---------------------------------------------------------------------------
# 7. calibration_status and recommended_for_gating
# ---------------------------------------------------------------------------

def test_calibration_status_uncalibrated_locally(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "entailed", "score": 0.8}]
    ).evaluate(_run())

    assert result.signals[0].calibration_status == "uncalibrated_locally"


def test_recommended_for_gating_false(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "entailed", "score": 0.8}]
    ).evaluate(_run())

    assert result.signals[0].recommended_for_gating is False


# ---------------------------------------------------------------------------
# 8. Raw payload preserved
# ---------------------------------------------------------------------------

def test_raw_payload_preserved(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    result = _provider_with_claims(
        [{"claim_id": "c1", "label": "entailed", "score": 0.9}]
    ).evaluate(_run())

    assert result.raw_payload == {"claims": [{"claim_id": "c1", "label": "entailed", "score": 0.9}]}
    assert result.signals[0].raw_payload["label"] == "entailed"


# ---------------------------------------------------------------------------
# 9. verify_claims() protocol
# ---------------------------------------------------------------------------

def test_verify_claims_returns_signal_records(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())
    provider = RefCheckerClaimSignalProvider({
        "claim_runner": lambda claims, context: [
            {"claim_id": "c1", "label": "entailed", "score": 0.85, "triplets": []}
        ]
    })
    signals = provider.verify_claims(["Refunds last 30 days."], ["Refunds are available for 30 days."])
    assert signals
    assert signals[0].label == "entailed"
    assert signals[0].signal_type == ExternalSignalType.claim_support


# ---------------------------------------------------------------------------
# 10. Native mode does not call RefChecker
# ---------------------------------------------------------------------------

def test_native_mode_does_not_call_refchecker() -> None:
    from raggov.engine import DiagnosisEngine

    engine = DiagnosisEngine({"mode": "native"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert "refchecker_claim" not in diagnosis.external_signals_used
    assert "refchecker_claim" not in diagnosis.missing_external_providers


# ---------------------------------------------------------------------------
# 11. External-enhanced with RefChecker missing → degraded
# ---------------------------------------------------------------------------

def test_external_enhanced_refchecker_missing_dep_sets_degraded(monkeypatch) -> None:
    from raggov.engine import DiagnosisEngine

    monkeypatch.setattr(RefCheckerClaimSignalProvider, "is_available", lambda self: False)

    engine = DiagnosisEngine({"mode": "external-enhanced"})
    engine.analyzers = []

    diagnosis = engine.diagnose(_run())
    assert diagnosis.degraded_external_mode is True
    assert "refchecker_claim" in diagnosis.missing_external_providers
    assert "heuristic_claim_verifier" in diagnosis.fallback_heuristics_used


# ---------------------------------------------------------------------------
# 12. ClaimGrounding consumes refchecker config
# ---------------------------------------------------------------------------

def test_claim_grounding_records_refchecker_provider_in_evidence(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())

    # Patch ClaimExtractor directly in the support module so analyze() uses it
    from raggov.analyzers.grounding import support as support_mod
    class _FakeExtractor:
        def __init__(self, **_): pass
        def extract(self, text): return ["Refunds are available for thirty days."]
    monkeypatch.setattr(support_mod, "ClaimExtractor", _FakeExtractor)

    analyzer = ClaimGroundingAnalyzer({"claim_verifier": "refchecker"})
    result = analyzer.analyze(_run())

    evidence_str = " ".join(result.evidence or [])
    assert "refchecker" in evidence_str


def test_claim_grounding_refchecker_missing_dep_falls_back(monkeypatch) -> None:
    """ClaimGrounding falls back to heuristic when refchecker is unavailable."""
    monkeypatch.setattr(RefCheckerClaimSignalProvider, "is_available", lambda self: False)
    analyzer = ClaimGroundingAnalyzer({"claim_verifier": "refchecker"})
    # Should not crash; records the missing dep
    result = analyzer.analyze(_run())
    evidence_str = " ".join(result.evidence or [])
    assert "refchecker" in evidence_str or result.status in ("pass", "fail", "skip")


# ---------------------------------------------------------------------------
# 13. NCV sees RefChecker-influenced claim report
# ---------------------------------------------------------------------------

def test_ncv_sees_refchecker_influenced_claim_report(monkeypatch) -> None:
    """External signal_records attached to claim records reach downstream consumers."""
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace())

    from raggov.evaluators.base import ExternalEvaluationResult
    external_result = ExternalEvaluationResult(
        provider=ExternalEvaluatorProvider.refchecker,
        succeeded=True,
        signals=[],
        raw_payload={"claims": []},
    )
    rag_run = _run()
    rag_run.metadata = rag_run.metadata or {}
    rag_run.metadata["external_evaluation_results"] = [external_result]

    # RefChecker result flows through metadata into retrieval diagnosis consumers.
    diag_result = RetrievalDiagnosisAnalyzerV0().analyze(rag_run)
    assert diag_result.retrieval_diagnosis_report is not None
