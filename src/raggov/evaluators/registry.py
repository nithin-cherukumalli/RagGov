"""External evaluator registry."""

from __future__ import annotations

import time
from typing import Any

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
)
from raggov.models.run import RAGRun


class ExternalEvaluatorRegistry:
    """Registry for external signal adapters.

    GovRAG remains responsible for diagnosis and final reports;
    registered adapters provide optional advisory signals only.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}

    def register(self, provider: Any) -> None:
        """Register an adapter under its .name attribute."""
        self._registry[provider.name] = provider

    def get(self, name_or_provider: str | ExternalEvaluatorProvider) -> Any | None:
        """Look up an adapter by name string or provider enum."""
        if isinstance(name_or_provider, ExternalEvaluatorProvider):
            for p in self._registry.values():
                if p.provider == name_or_provider:
                    return p
            return None
        return self._registry.get(name_or_provider)

    def list_registered(self) -> list[str]:
        """Return names of all registered adapters."""
        return list(self._registry.keys())

    def list_available(self) -> list[str]:
        """Return names of adapters whose dependencies are satisfied."""
        return [name for name, p in self._registry.items() if p.is_available()]

    def evaluate_enabled(
        self,
        run: RAGRun,
        enabled_providers: list[str | ExternalEvaluatorProvider],
        strict_mode: bool = False,
    ) -> list[ExternalEvaluationResult]:
        """Run each enabled adapter and collect results.

        Rules:
        - Missing dependency → succeeded=False, missing_dependency=True (never silent).
        - Adapter exception in non-strict mode → succeeded=False, error captured.
        - strict_mode=True → adapter exception propagates to caller.
        """
        results: list[ExternalEvaluationResult] = []

        for key in enabled_providers:
            adapter = self.get(key)
            if adapter is None:
                if strict_mode:
                    raise ValueError(f"Unknown external evaluator provider: {key}")
                # We skip 'a2p' here because it is a known native analyzer, not an external adapter.
                # All other unknowns are recorded as adapter errors.
                if key != "a2p":
                    results.append(
                        ExternalEvaluationResult(
                            provider=ExternalEvaluatorProvider.custom,
                            adapter_name=str(key),
                            succeeded=False,
                            error=f"Unknown external evaluator provider: {key}",
                        )
                    )
                continue

            if not adapter.is_available():
                results.append(
                    ExternalEvaluationResult(
                        provider=adapter.provider,
                        adapter_name=adapter.name,
                        succeeded=False,
                        missing_dependency=True,
                        error=f"Adapter '{adapter.name}' dependency not available.",
                    )
                )
                continue

            t0 = time.monotonic()
            try:
                result = adapter.evaluate(run)
                result.adapter_name = adapter.name
                result.latency_ms = (time.monotonic() - t0) * 1000
                results.append(result)
            except Exception as exc:
                if strict_mode:
                    raise
                results.append(
                    ExternalEvaluationResult(
                        provider=adapter.provider,
                        adapter_name=adapter.name,
                        succeeded=False,
                        error=str(exc),
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                )

        return results


def create_standard_registry(config: dict[str, Any] | None = None) -> ExternalEvaluatorRegistry:
    """Create a registry and register all standard optional external adapters.

    Adapters are initialized with the provided config (e.g. llm_client).
    """
    from raggov.evaluators.citation.refchecker_adapter import RefCheckerCitationAdapter
    from raggov.evaluators.citation.structured_llm import StructuredLLMCitationVerifierAdapter
    from raggov.evaluators.claim.refchecker_adapter import RefCheckerClaimAdapter
    from raggov.evaluators.claim.structured_llm import StructuredLLMClaimVerifierAdapter
    from raggov.evaluators.retrieval.cross_encoder import CrossEncoderRetrievalRelevanceProvider
    from raggov.evaluators.retrieval.deepeval_adapter import DeepEvalAdapter
    from raggov.evaluators.retrieval.ragas_adapter import RAGASAdapter
    from raggov.evaluators.retrieval.ragchecker_adapter import RAGCheckerSignalProvider

    registry = ExternalEvaluatorRegistry()
    registry.register(StructuredLLMClaimVerifierAdapter(config))
    registry.register(StructuredLLMCitationVerifierAdapter(config))
    registry.register(DeepEvalAdapter(config))
    registry.register(RAGASAdapter(config))
    registry.register(CrossEncoderRetrievalRelevanceProvider(config))
    registry.register(RefCheckerClaimAdapter(config))
    registry.register(RefCheckerCitationAdapter(config))
    registry.register(RAGCheckerSignalProvider(config))
    return registry
