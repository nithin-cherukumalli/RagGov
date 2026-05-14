"""Optional DeepEval retrieval/context signal adapter.

DeepEval is an optional dependency. Install via: pip install govrag[eval] or pip install deepeval

Input mapping from RAGRun:
  input / query         = run.query
  actual_output / answer= run.final_answer
  retrieval_context     = [chunk.text for chunk in run.retrieved_chunks]
  expected_output       = run.metadata.get("reference_answer")  (optional gold answer)

If expected_output (reference) is missing:
  contextual_recall requires reference — a visible ExternalSignalRecord with
  label="missing_reference_input" is emitted instead of silently skipping.
  contextual_precision, contextual_relevancy, and faithfulness may still run.

Signal label thresholds (uncalibrated, configurable):
  >= 0.75  → high
  0.40–0.75 → medium
  < 0.40   → low
  None / error → unavailable
"""

from __future__ import annotations

import importlib
from typing import Any

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.models.run import RAGRun


_DEEPEVAL_METRICS: dict[str, tuple[str, ExternalSignalType, str]] = {
    "contextual_relevancy": (
        "deepeval_contextual_relevancy",
        ExternalSignalType.retrieval_contextual_relevancy,
        "Low contextual relevancy is advisory evidence for retrieval noise.",
    ),
    "contextual_precision": (
        "deepeval_contextual_precision",
        ExternalSignalType.retrieval_contextual_precision,
        "Poor contextual precision is advisory evidence for rank failure.",
    ),
    "context_recall": (
        "deepeval_context_recall",
        ExternalSignalType.retrieval_context_recall,
        "Low context recall is advisory evidence for retrieval miss.",
    ),
    "faithfulness": (
        "deepeval_faithfulness",
        ExternalSignalType.faithfulness,
        "Faithfulness is advisory grounding evidence and must not override GovRAG claim verification.",
    ),
}

# Metrics that require a reference/gold answer to run.
_REFERENCE_REQUIRED_METRICS: frozenset[str] = frozenset({"context_recall"})

# Default derived-label thresholds (uncalibrated).
_DEFAULT_HIGH_THRESHOLD: float = 0.75
_DEFAULT_LOW_THRESHOLD: float = 0.40


def _derive_label(value: Any, *, high: float, low: float) -> str:
    """Map a numeric metric value to a human-readable label."""
    if not isinstance(value, (int, float)):
        return "unavailable"
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


class DeepEvalRetrievalSignalProvider:
    """Provide advisory DeepEval retrieval/context metrics when available.

    This adapter:
    - Is fully optional; missing deepeval returns succeeded=False, missing_dependency=True.
    - Emits ExternalSignalRecord objects with calibration_status="uncalibrated_locally"
      and recommended_for_gating=False.
    - Derives human-readable labels (high/medium/low) using configurable thresholds.
    - Surfaces missing reference inputs for metrics that require them.
    - Never directly sets a GovRAG FailureType. GovRAG owns all diagnosis decisions.
    """

    name: str = "deepeval"
    provider: ExternalEvaluatorProvider = ExternalEvaluatorProvider.deepeval

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._high_threshold: float = float(
            self.config.get("deepeval_label_high_threshold", _DEFAULT_HIGH_THRESHOLD)
        )
        self._low_threshold: float = float(
            self.config.get("deepeval_label_low_threshold", _DEFAULT_LOW_THRESHOLD)
        )

    def is_available(self) -> bool:
        try:
            importlib.import_module("deepeval")
            return True
        except ImportError:
            return False

    def check_readiness(self) -> ProviderReadiness:
        from raggov.evaluators.readiness import ProviderReadiness, package_version_or_none
        enabled = set(self.config.get("enabled_external_providers", []))
        if enabled and self.name not in enabled:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="disabled",
                reason_code="disabled",
                reason="Provider is not enabled in the current configuration.",
                fallback_provider="native_retrieval_signals_only",
            )

        package_version = package_version_or_none("deepeval")
        if not self.is_available():
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="package_missing",
                reason="DeepEval package could not be imported.",
                install_hint="pip install deepeval",
                fallback_provider="native_retrieval_signals_only",
                package_version=package_version,
            )

        if self.config.get("deepeval_metric_results") is not None or self.config.get("metric_results") is not None:
            maturity = "mock_runner"
            runtime_available = True
            runtime_reason = "Using mock metric_results."
        elif self.config.get("deepeval_metric_runner") is not None or self.config.get("metric_runner") is not None:
            maturity = "configured_runner"
            runtime_available = True
            runtime_reason = "Using user-configured runner."
        else:
            maturity = "schema_only"
            runtime_available = False
            runtime_reason = "No runner or mock results configured. Native runtime execution not implemented."

        if not runtime_available:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="degraded",
                reason_code="runtime_execution_not_configured",
                reason=runtime_reason,
                fallback_provider="native_retrieval_signals_only",
                package_version=package_version,
                integration_maturity=maturity,
                runtime_execution_available=runtime_available,
                runtime_execution_reason=runtime_reason,
            )

        return ProviderReadiness(
            provider_name=self.name,
            available=True,
            status="available",
            fallback_provider="native_retrieval_signals_only",
            package_version=package_version,
            integration_maturity=maturity,
            runtime_execution_available=runtime_available,
            runtime_execution_reason=runtime_reason,
        )

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        """Evaluate DeepEval metrics for the given RAGRun."""
        if not self.is_available():
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                missing_dependency=True,
                error=(
                    "deepeval: package not installed. "
                    "Install optional extra `govrag[eval]` or `pip install deepeval`."
                ),
            )

        try:
            payload = self._metric_payload(run)
        except Exception as exc:
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                error=f"deepeval evaluation failed: {exc}",
            )

        # Determine whether a reference answer is available.
        reference = run.metadata.get("reference_answer") if run.metadata else None

        signals: list[ExternalSignalRecord] = []

        # Check for metrics that require reference but reference is missing.
        for metric in _REFERENCE_REQUIRED_METRICS:
            if metric in payload and reference is None:
                signals.append(
                    ExternalSignalRecord(
                        provider=self.provider,
                        signal_type=ExternalSignalType.custom,
                        metric_name=f"deepeval_{metric}_missing_reference",
                        value=None,
                        label="missing_reference_input",
                        explanation=(
                            f"DeepEval metric '{metric}' requires a reference/gold answer "
                            "which is absent from run.metadata['reference_answer']. "
                            "Result may be unreliable or skipped by DeepEval."
                        ),
                        raw_payload={"metric": metric, "reference_present": False},
                        method_type="external_signal_adapter",
                        calibration_status="uncalibrated_locally",
                        recommended_for_gating=False,
                        limitations=[
                            "missing_reference_input_for_metric",
                            "deepeval_signal_is_advisory_not_diagnostic",
                        ],
                    )
                )

        # Emit a signal for each metric present in the payload.
        for metric, value in payload.items():
            if metric not in _DEEPEVAL_METRICS:
                continue
            signals.append(self._signal(metric, value))

        return ExternalEvaluationResult(
            provider=self.provider,
            succeeded=True,
            signals=signals,
            raw_payload=dict(payload),
        )

    def score_relevance(
        self, query: str, chunks: list[str]
    ) -> list[ExternalSignalRecord]:
        payload = self.config.get("metric_runner", lambda _run: {})(
            {"query": query, "chunks": chunks}
        )
        return [
            self._signal(metric, value)
            for metric, value in dict(payload).items()
            if metric in _DEEPEVAL_METRICS
        ]

    def _metric_payload(self, run: RAGRun) -> dict[str, Any]:
        runner = self.config.get("deepeval_metric_runner") or self.config.get("metric_runner")
        if runner is not None:
            return dict(runner(run))

        configured = self.config.get("deepeval_metric_results") or self.config.get("metric_results")
        if configured is not None:
            return dict(configured)

        return {}

    def _signal(self, metric: str, value: Any) -> ExternalSignalRecord:
        metric_name, signal_type, interpretation = _DEEPEVAL_METRICS[metric]
        label = _derive_label(
            value,
            high=self._high_threshold,
            low=self._low_threshold,
        )
        return ExternalSignalRecord(
            provider=self.provider,
            signal_type=signal_type,
            metric_name=metric_name,
            value=value,
            label=label,
            explanation=interpretation,
            raw_payload={"metric": metric, "value": value},
            method_type="external_signal_adapter",
            calibration_status="uncalibrated_locally",
            recommended_for_gating=False,
            limitations=[
                f"label_thresholds_uncalibrated: high>={self._high_threshold}, low<{self._low_threshold}",
                "deepeval_signal_is_advisory_not_diagnostic",
            ],
        )


# Backward-compatible import name for the previous skeleton.
DeepEvalAdapter = DeepEvalRetrievalSignalProvider
