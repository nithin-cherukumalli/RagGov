"""Optional RAGChecker diagnostic signal adapter.

RAGChecker provides comprehensive RAG evaluation metrics covering both retrieval
and generation quality (e.g., context precision, claim recall, faithfulness).

Input mapping from RAGRun:
  query              = run.query
  response           = run.final_answer
  retrieved_context  = [chunk.text for chunk in run.retrieved_chunks]
  reference          = run.metadata.get("reference_answer") or run.metadata.get("gold_answer")
  gold_claims        = run.metadata.get("gold_claims")

If required inputs like reference answers or gold claims are missing, the adapter
will emit visible 'missing_reference_input' signals instead of silently skipping them.

All emitted signals are treated as advisory evidence (calibration_status="uncalibrated_locally").
They do not unilaterally determine GovRAG's final FailureType.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.evaluators.readiness import (
    ProviderReadiness,
    package_version_or_none,
)
from raggov.models.run import RAGRun


_RAGCHECKER_METRICS: dict[str, tuple[str, ExternalSignalType, str]] = {
    "context_precision": (
        "ragchecker_context_precision",
        ExternalSignalType.retrieval_context_precision,
        "Low context precision is advisory evidence for retrieval noise.",
    ),
    "context_utilization": (
        "ragchecker_context_utilization",
        ExternalSignalType.context_utilization,
        "Low context utilization suggests retrieved evidence may have been ignored by generation.",
    ),
    "retrieval_context_recall": (
        "ragchecker_retrieval_context_recall",
        ExternalSignalType.retrieval_context_recall,
        "Low retrieval context recall is advisory evidence for retrieval miss.",
    ),
    "claim_recall": (
        "ragchecker_claim_recall",
        ExternalSignalType.claim_recall,
        "Low claim recall is advisory evidence for retrieval miss when sufficiency/grounding also indicate missing support.",
    ),
    "faithfulness": (
        "ragchecker_faithfulness",
        ExternalSignalType.faithfulness,
        "Faithfulness is advisory grounding evidence; does not alone force a diagnosis.",
    ),
    "hallucination": (
        "ragchecker_hallucination",
        ExternalSignalType.hallucination,
        "High hallucination is advisory evidence for claim support/generation issues, not retrieval failure alone.",
    ),
    "claim_support": (
        "ragchecker_claim_support",
        ExternalSignalType.claim_support,
        "Advisory evidence for claim support.",
    ),
}

# Metrics that require reference/gold answer to be meaningful.
_REFERENCE_REQUIRED_METRICS: frozenset[str] = frozenset({
    "claim_recall",
    "retrieval_context_recall",
})

_DEFAULT_HIGH_THRESHOLD: float = 0.75
_DEFAULT_LOW_THRESHOLD: float = 0.40


def _derive_label(value: Any, *, high: float, low: float, metric: str) -> str:
    """Map a numeric metric value to a human-readable label.
    
    Note: For hallucination, lower is better. For others, higher is better.
    """
    if not isinstance(value, (int, float)):
        return "unavailable"
        
    if "hallucination" in metric.lower():
        if value <= (1.0 - high):
            return "high"  # High quality (low hallucination)
        if value <= (1.0 - low):
            return "medium"
        return "low"
        
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


class RAGCheckerSignalProvider:
    """Provide advisory RAGChecker metrics when the package is installed.

    - Is fully optional; missing RAGChecker returns missing_dependency=True.
    - Emits ExternalSignalRecord objects with calibration_status="uncalibrated_locally".
    - Derives human-readable labels (high/medium/low).
    - Surfaces missing reference inputs for metrics that require them.
    - Never directly sets a GovRAG FailureType.
    """

    name: str = "ragchecker"
    provider: ExternalEvaluatorProvider = ExternalEvaluatorProvider.ragchecker

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._high_threshold: float = float(
            self.config.get("ragchecker_label_high_threshold", _DEFAULT_HIGH_THRESHOLD)
        )
        self._low_threshold: float = float(
            self.config.get("ragchecker_label_low_threshold", _DEFAULT_LOW_THRESHOLD)
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("ragchecker") is not None

    def check_readiness(self) -> ProviderReadiness:
        enabled = set(self.config.get("enabled_external_providers", []))
        if enabled and self.name not in enabled:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="disabled",
                reason_code="disabled",
                reason="Provider is not enabled in the current configuration.",
                fallback_provider="native_context_metrics",
            )

        package_version = package_version_or_none("ragchecker")
        import_name = self.config.get("ragchecker_import_name", "ragchecker")
        try:
            module = importlib.import_module(import_name)
        except ImportError:
            code = "package_missing" if import_name == "ragchecker" else "import_path_mismatch"
            reason = (
                "RAGChecker package/module could not be imported."
                if code == "package_missing"
                else f"Configured ragchecker import path '{import_name}' could not be imported."
            )
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code=code,
                reason=reason,
                install_hint="pip install ragchecker",
                fallback_provider="native_context_metrics",
                package_version=package_version,
            )
        except Exception as exc:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="unknown_import_error",
                reason=f"Unexpected RAGChecker import error: {exc}",
                config_hint="Verify the installed ragchecker package matches the adapter expectations.",
                fallback_provider="native_context_metrics",
                package_version=package_version,
            )

        required_attr = self.config.get("ragchecker_required_attr")
        if required_attr and not hasattr(module, required_attr):
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="adapter_api_mismatch",
                reason=f"Installed ragchecker package does not expose required attribute '{required_attr}'.",
                config_hint="Adjust ragchecker_required_attr or install a compatible ragchecker package build.",
                fallback_provider="native_context_metrics",
                package_version=package_version,
            )

        llm_mode = self.config.get("ragchecker_mode") == "llm"
        has_llm = bool(self.config.get("llm_client")) or bool(self.config.get("llm_fn"))
        if llm_mode and not has_llm:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="llm_config_missing",
                reason="RAGChecker is configured for LLM-backed evaluation but no llm_client/llm_fn is configured.",
                config_hint="Provide llm_client or llm_fn, or switch ragchecker_mode away from llm.",
                requires_network=True,
                safe_offline=False,
                fallback_provider="native_context_metrics",
                package_version=package_version,
            )

        if self.config.get("ragchecker_metric_results") is not None or self.config.get("metric_results") is not None:
            maturity = "mock_runner"
            runtime_available = True
            runtime_reason = "Using mock metric_results."
        elif self.config.get("ragchecker_metric_runner") is not None or self.config.get("metric_runner") is not None:
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
                fallback_provider="native_context_metrics",
                package_version=package_version,
                integration_maturity=maturity,
                runtime_execution_available=runtime_available,
                runtime_execution_reason=runtime_reason,
            )

        return ProviderReadiness(
            provider_name=self.name,
            available=True,
            status="available",
            fallback_provider="native_context_metrics",
            package_version=package_version,
            integration_maturity=maturity,
            runtime_execution_available=runtime_available,
            runtime_execution_reason=runtime_reason,
        )

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        if not self.is_available():
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                missing_dependency=True,
                error=(
                    "ragchecker: package not installed. "
                    "Install optional extra `govrag[eval]` or `pip install ragchecker`."
                ),
            )

        try:
            payload = self._metric_payload(run)
        except Exception as exc:
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                error=f"ragchecker evaluation failed: {exc}",
            )

        # Reference answers/claims extraction
        reference = None
        if run.metadata:
            reference = run.metadata.get("reference_answer") or run.metadata.get("gold_answer")
            gold_claims = run.metadata.get("gold_claims")
            if reference is None and gold_claims is not None:
                # If we have gold claims but no reference answer, we might still be able to do some recall
                reference = True  # acts as a flag that reference material exists

        signals: list[ExternalSignalRecord] = []

        # Handle metrics requiring reference
        for metric in _REFERENCE_REQUIRED_METRICS:
            if metric in payload and reference is None:
                signals.append(
                    ExternalSignalRecord(
                        provider=self.provider,
                        signal_type=ExternalSignalType.custom,
                        metric_name=f"ragchecker_{metric}_missing_reference",
                        value=None,
                        label="missing_reference_input",
                        explanation=(
                            f"RAGChecker metric '{metric}' requires a reference answer "
                            "or gold claims which are absent from run.metadata. "
                            "Result may be unreliable or skipped."
                        ),
                        raw_payload={"metric": metric, "reference_present": False},
                        method_type="external_signal_adapter",
                        calibration_status="uncalibrated_locally",
                        recommended_for_gating=False,
                        limitations=[
                            "missing_reference_input_for_metric",
                            "ragchecker_signal_is_advisory_not_diagnostic",
                        ],
                    )
                )

        # Process valid metrics
        for metric, value in payload.items():
            if metric not in _RAGCHECKER_METRICS:
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
            if metric in _RAGCHECKER_METRICS
        ]

    def _metric_payload(self, run: RAGRun) -> dict[str, Any]:
        runner = self.config.get("ragchecker_metric_runner") or self.config.get("metric_runner")
        if runner is not None:
            return dict(runner(run))

        configured = self.config.get("ragchecker_metric_results") or self.config.get("metric_results")
        if configured is not None:
            return dict(configured)

        return {}

    def _signal(self, metric: str, value: Any) -> ExternalSignalRecord:
        metric_name, signal_type, interpretation = _RAGCHECKER_METRICS[metric]
        label = _derive_label(
            value,
            high=self._high_threshold,
            low=self._low_threshold,
            metric=metric,
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
                "RAGChecker signal is not locally calibrated for this corpus.",
                "Metric output should be treated as advisory evidence.",
                "GovRAG uses RAGChecker output as evidence, not final diagnosis.",
            ],
        )
