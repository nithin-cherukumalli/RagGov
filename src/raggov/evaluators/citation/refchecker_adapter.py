"""Optional RefChecker citation verification signal adapter.

RefChecker is an optional dependency. Install via: pip install refchecker
(or as documented in the project — the package may require manual install).

Input mapping from GovRAG:
  claim_text    = claim text
  cited_doc_id  = source doc being cited
  cited_text    = retrieved chunk text from that source
  context       = full retrieved context list

Citation label normalization:
  supports           → citation text supports the claim
  contradicts        → citation text contradicts the claim
  does_not_support   → citation is present but does not support
  citation_missing   → no suitable citation found
  unclear            → cannot determine

"unclear" is never treated as support.
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
from raggov.evaluators.readiness import (
    ProviderReadiness,
    package_version_or_none,
)
from raggov.models.run import RAGRun


_VALID_CITATION_LABELS: frozenset[str] = frozenset(
    {"supports", "contradicts", "does_not_support", "citation_missing", "unclear"}
)

_DEFAULT_HIGH_THRESHOLD: float = 0.75
_DEFAULT_LOW_THRESHOLD: float = 0.40


def _normalize_citation_label(raw: str | None) -> str:
    """Normalize a raw RefChecker citation label to a canonical GovRAG label."""
    if raw is None:
        return "unclear"
    mapped = {
        # entailment family → supports
        "entailment": "supports",
        "entailed": "supports",
        "supported": "supports",
        "supports": "supports",
        # contradiction family
        "contradiction": "contradicts",
        "contradicted": "contradicts",
        "contradicts": "contradicts",
        # neutral / no-support family
        "neutral": "does_not_support",
        "unsupported": "does_not_support",
        "does_not_support": "does_not_support",
        "hallucinated": "does_not_support",
        # missing
        "missing": "citation_missing",
        "citation_missing": "citation_missing",
        # unclear
        "unknown": "unclear",
        "unclear": "unclear",
    }
    return mapped.get(raw.lower().strip(), "unclear")


class RefCheckerCitationSignalProvider:
    """Provide advisory RefChecker citation verification signals when available.

    This adapter:
    - Is fully optional; missing refchecker returns succeeded=False, missing_dependency=True.
    - Emits ExternalSignalRecord with calibration_status="uncalibrated_locally"
      and recommended_for_gating=False.
    - Normalizes citation labels (supports/contradicts/does_not_support/citation_missing/unclear).
    - Respects per-claim citation mapping; does not smear global citation IDs.
    - Never directly sets a GovRAG FailureType. GovRAG owns all diagnosis.
    """

    name: str = "refchecker_citation"
    provider: ExternalEvaluatorProvider = ExternalEvaluatorProvider.refchecker

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._high_threshold: float = float(
            self.config.get("refchecker_label_high_threshold", _DEFAULT_HIGH_THRESHOLD)
        )
        self._low_threshold: float = float(
            self.config.get("refchecker_label_low_threshold", _DEFAULT_LOW_THRESHOLD)
        )

    def is_available(self) -> bool:
        try:
            importlib.import_module("refchecker")
            return True
        except ImportError:
            return False

    def check_readiness(self) -> ProviderReadiness:
        enabled = set(self.config.get("enabled_external_providers", []))
        if enabled and self.name not in enabled:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="disabled",
                reason_code="disabled",
                reason="Provider is not enabled in the current configuration.",
                fallback_provider="native_citation_verifier",
            )

        package_version = package_version_or_none("refchecker")
        try:
            module = importlib.import_module("refchecker")
        except ImportError:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="package_missing",
                reason="RefChecker package/module could not be imported.",
                install_hint="pip install refchecker",
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )
        except Exception as exc:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="unknown_import_error",
                reason=f"Unexpected RefChecker import error: {exc}",
                install_hint='pip install "refchecker[open-extractor,repcex]"',
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )

        try:
            importlib.import_module("spacy")
        except ImportError:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="spacy_missing",
                reason="spaCy is required by the RefChecker adapter but is not installed.",
                install_hint="pip install spacy",
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )

        try:
            importlib.import_module("en_core_web_sm")
        except ImportError:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="spacy_model_missing",
                reason="spaCy model en_core_web_sm is not installed.",
                install_hint="python -m spacy download en_core_web_sm",
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )

        required_attr = self.config.get("refchecker_required_attr")
        if required_attr and not hasattr(module, required_attr):
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="adapter_api_mismatch",
                reason=f"Installed refchecker package does not expose required attribute '{required_attr}'.",
                config_hint="Verify the installed refchecker package version matches the adapter expectations.",
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )

        llm_mode = self.config.get("refchecker_mode") == "llm"
        has_llm = bool(self.config.get("llm_client")) or bool(self.config.get("llm_fn"))
        if llm_mode and not has_llm:
            return ProviderReadiness(
                provider_name=self.name,
                available=False,
                status="unavailable",
                reason_code="llm_config_missing",
                reason="RefChecker is configured for LLM-backed checking but no llm_client/llm_fn is configured.",
                config_hint="Provide llm_client or llm_fn, or switch refchecker_mode away from llm.",
                requires_network=True,
                safe_offline=False,
                fallback_provider="native_citation_verifier",
                package_version=package_version,
            )

        if self.config.get("refchecker_citation_metric_results") is not None or self.config.get("metric_results") is not None:
            maturity = "mock_runner"
            runtime_available = True
            runtime_reason = "Using mock metric_results."
        elif self.config.get("refchecker_citation_metric_runner") is not None or self.config.get("metric_runner") is not None or self.config.get("citation_runner") is not None:
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
                fallback_provider="native_citation_verifier",
                package_version=package_version,
                integration_maturity=maturity,
                runtime_execution_available=runtime_available,
                runtime_execution_reason=runtime_reason,
            )

        return ProviderReadiness(
            provider_name=self.name,
            available=True,
            status="available",
            fallback_provider="native_citation_verifier",
            package_version=package_version,
            integration_maturity=maturity,
            runtime_execution_available=runtime_available,
            runtime_execution_reason=runtime_reason,
        )

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        """Evaluate RefChecker citation signals for the given RAGRun."""
        if not self.is_available():
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                missing_dependency=True,
                error=(
                    "refchecker: package not installed. "
                    "Install via `pip install refchecker` or consult project docs."
                ),
            )

        try:
            payload = self._metric_payload(run)
        except Exception as exc:
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                error=f"refchecker citation evaluation failed: {exc}",
            )

        signals = self._signals_from_payload(payload)
        return ExternalEvaluationResult(
            provider=self.provider,
            succeeded=True,
            signals=signals,
            raw_payload=dict(payload),
        )

    def verify_citations(
        self, cited_ids: list[str], chunks: list[str]
    ) -> list[ExternalSignalRecord]:
        """Verify citation texts against claim context.

        Returns one ExternalSignalRecord per citation.
        Uses citation_runner if configured; otherwise returns empty list.
        """
        runner = self.config.get("citation_runner")
        if runner is None:
            return []
        results = runner(cited_ids, chunks)
        signals = []
        for i, item in enumerate(results or []):
            claim_id = item.get("claim_id", f"claim_{i}")
            cited_id = item.get("cited_id", "")
            raw_label = item.get("label")
            label = _normalize_citation_label(raw_label)
            score = item.get("score")
            signals.append(
                self._build_signal(
                    claim_id=claim_id,
                    cited_id=cited_id,
                    raw_label=raw_label,
                    label=label,
                    score=score,
                    triplets=item.get("triplets", []),
                )
            )
        return signals

    def _metric_payload(self, run: RAGRun) -> dict[str, Any]:
        runner = self.config.get("refchecker_citation_metric_runner") or self.config.get("metric_runner")
        if runner is not None:
            return dict(runner(run))
        configured = self.config.get("refchecker_citation_metric_results") or self.config.get("metric_results")
        if configured is not None:
            return dict(configured)
        return {}

    def _signals_from_payload(self, payload: dict[str, Any]) -> list[ExternalSignalRecord]:
        citations_data = payload.get("citations", [])
        return [
            self._build_signal(
                claim_id=item.get("claim_id", "unknown"),
                cited_id=item.get("cited_id", ""),
                raw_label=item.get("label"),
                label=_normalize_citation_label(item.get("label")),
                score=item.get("score"),
                triplets=item.get("triplets", []),
            )
            for item in citations_data
        ]

    def _build_signal(
        self,
        *,
        claim_id: str,
        cited_id: str,
        raw_label: str | None,
        label: str,
        score: Any,
        triplets: list,
    ) -> ExternalSignalRecord:
        return ExternalSignalRecord(
            provider=self.provider,
            signal_type=ExternalSignalType.citation_support,
            metric_name="refchecker_citation_check",
            value=score,
            label=label,
            explanation=(
                f"RefChecker citation check: claim={claim_id}, cited={cited_id}, label={label}. "
                "Advisory evidence only — GovRAG owns final citation diagnosis."
            ),
            affected_claim_ids=[claim_id],
            affected_doc_ids=[cited_id] if cited_id else [],
            raw_payload={"label": raw_label, "score": score, "triplets": triplets},
            method_type="external_signal_adapter",
            calibration_status="uncalibrated_locally",
            recommended_for_gating=False,
            limitations=[
                "refchecker_signal_is_advisory_not_diagnostic",
                "uncalibrated_locally",
                "unclear_label_is_not_treated_as_support",
            ],
        )


# Backward-compatible alias.
RefCheckerCitationAdapter = RefCheckerCitationSignalProvider
