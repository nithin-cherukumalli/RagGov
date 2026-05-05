"""Tests for ExternalEvaluatorRegistry."""

from __future__ import annotations

import pytest

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.evaluators.registry import ExternalEvaluatorRegistry
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run() -> RAGRun:
    return RAGRun(
        query="What is the capital of France?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                source_doc_id="doc-1",
                text="Paris is the capital of France.",
                score=0.95,
            )
        ],
        final_answer="Paris.",
    )


class _AlwaysAvailableAdapter:
    name = "always_available"
    provider = ExternalEvaluatorProvider.custom

    def is_available(self) -> bool:
        return True

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        signal = ExternalSignalRecord(
            provider=self.provider,
            signal_type=ExternalSignalType.custom,
            metric_name="always_available_metric",
            value=1.0,
            raw_payload={"source": "always_available"},
        )
        return ExternalEvaluationResult(
            provider=self.provider,
            succeeded=True,
            signals=[signal],
            raw_payload={"run_id": run.run_id},
        )


class _NeverAvailableAdapter:
    name = "never_available"
    provider = ExternalEvaluatorProvider.ragas

    def is_available(self) -> bool:
        return False

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        return ExternalEvaluationResult(
            provider=self.provider,
            succeeded=False,
            missing_dependency=True,
            error="ragas not installed.",
        )


class _ExplodingAdapter:
    name = "exploding"
    provider = ExternalEvaluatorProvider.deepeval

    def is_available(self) -> bool:
        return True

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        raise RuntimeError("DeepEval network timeout")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_and_list_registered() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    assert "always_available" in registry.list_registered()


def test_register_multiple_adapters() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    registry.register(_NeverAvailableAdapter())
    registered = registry.list_registered()
    assert "always_available" in registered
    assert "never_available" in registered


def test_get_by_name() -> None:
    registry = ExternalEvaluatorRegistry()
    adapter = _AlwaysAvailableAdapter()
    registry.register(adapter)
    result = registry.get("always_available")
    assert result is adapter


def test_get_by_provider_enum() -> None:
    registry = ExternalEvaluatorRegistry()
    adapter = _AlwaysAvailableAdapter()
    registry.register(adapter)
    result = registry.get(ExternalEvaluatorProvider.custom)
    assert result is adapter


def test_get_unknown_returns_none() -> None:
    registry = ExternalEvaluatorRegistry()
    assert registry.get("nonexistent") is None


def test_get_by_provider_enum_not_registered_returns_none() -> None:
    registry = ExternalEvaluatorRegistry()
    assert registry.get(ExternalEvaluatorProvider.nli) is None


# ---------------------------------------------------------------------------
# list_available
# ---------------------------------------------------------------------------


def test_list_available_excludes_unavailable() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    registry.register(_NeverAvailableAdapter())
    available = registry.list_available()
    assert "always_available" in available
    assert "never_available" not in available


def test_list_available_empty_when_all_unavailable() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_NeverAvailableAdapter())
    assert registry.list_available() == []


# ---------------------------------------------------------------------------
# Missing dependency behavior
# ---------------------------------------------------------------------------


def test_evaluate_enabled_missing_dependency_surfaces_error() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_NeverAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["never_available"])

    assert len(results) == 1
    result = results[0]
    assert result.succeeded is False
    assert result.missing_dependency is True
    assert result.error is not None


def test_evaluate_enabled_missing_dependency_not_silent() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_NeverAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["never_available"])
    assert results[0].error is not None and len(results[0].error) > 0


# ---------------------------------------------------------------------------
# Non-strict exception capture
# ---------------------------------------------------------------------------


def test_evaluate_enabled_captures_exception_non_strict() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_ExplodingAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["exploding"], strict_mode=False)

    assert len(results) == 1
    result = results[0]
    assert result.succeeded is False
    assert result.missing_dependency is False
    assert "timeout" in (result.error or "").lower()


def test_evaluate_enabled_non_strict_does_not_crash_pipeline() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    registry.register(_ExplodingAdapter())
    registry.register(_NeverAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(
        run,
        ["always_available", "exploding", "never_available"],
        strict_mode=False,
    )

    assert len(results) == 3
    successes = [r for r in results if r.succeeded]
    failures = [r for r in results if not r.succeeded]
    assert len(successes) == 1
    assert len(failures) == 2


# ---------------------------------------------------------------------------
# Strict mode raises
# ---------------------------------------------------------------------------


def test_evaluate_enabled_strict_mode_raises() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_ExplodingAdapter())
    run = _make_run()

    with pytest.raises(RuntimeError, match="timeout"):
        registry.evaluate_enabled(run, ["exploding"], strict_mode=True)


# ---------------------------------------------------------------------------
# Calibration and gating invariants
# ---------------------------------------------------------------------------


def test_all_signals_have_uncalibrated_locally() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["always_available"])
    for result in results:
        for signal in result.signals:
            assert signal.calibration_status == "uncalibrated_locally"


def test_all_signals_have_recommended_for_gating_false() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["always_available"])
    for result in results:
        for signal in result.signals:
            assert signal.recommended_for_gating is False


# ---------------------------------------------------------------------------
# raw_payload preserved
# ---------------------------------------------------------------------------


def test_raw_payload_preserved_in_result() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["always_available"])
    assert results[0].raw_payload is not None
    assert results[0].raw_payload.get("run_id") == run.run_id


def test_raw_payload_preserved_in_signals() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["always_available"])
    signal = results[0].signals[0]
    assert signal.raw_payload == {"source": "always_available"}


# ---------------------------------------------------------------------------
# evaluate_enabled with unknown provider skips silently
# ---------------------------------------------------------------------------


def test_evaluate_enabled_records_unregistered_provider() -> None:
    registry = ExternalEvaluatorRegistry()
    run = _make_run()

    results = registry.evaluate_enabled(run, ["nonexistent_adapter"])
    assert len(results) == 1
    assert results[0].succeeded is False
    assert "Unknown" in results[0].error
    assert results[0].adapter_name == "nonexistent_adapter"


def test_evaluate_enabled_strict_mode_unknown_provider_raises() -> None:
    registry = ExternalEvaluatorRegistry()
    run = _make_run()

    with pytest.raises(ValueError, match="Unknown external evaluator provider"):
        registry.evaluate_enabled(run, ["nonexistent_adapter"], strict_mode=True)


def test_evaluate_enabled_latency_populated_on_success() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_AlwaysAvailableAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["always_available"])
    assert results[0].latency_ms is not None
    assert results[0].latency_ms >= 0


def test_evaluate_enabled_latency_populated_on_exception() -> None:
    registry = ExternalEvaluatorRegistry()
    registry.register(_ExplodingAdapter())
    run = _make_run()

    results = registry.evaluate_enabled(run, ["exploding"], strict_mode=False)
    assert results[0].latency_ms is not None
    assert results[0].latency_ms >= 0
