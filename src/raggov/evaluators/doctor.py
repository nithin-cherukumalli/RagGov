"""Doctor layer for external provider readiness."""

from __future__ import annotations

from typing import Any

from raggov.evaluators.readiness import (
    ExternalProviderDoctorReport,
    ProviderReadiness,
)


def build_provider_doctor_report(
    registry: Any,
    *,
    enabled_providers: list[str] | None = None,
) -> ExternalProviderDoctorReport:
    """Collect readiness for all registered providers.

    Providers that do not implement ``check_readiness`` fall back to a coarse
    availability probe so the doctor remains useful while adapter coverage is
    incremental.
    """

    providers: list[ProviderReadiness] = []
    available: list[str] = []
    unavailable: list[str] = []
    degraded: list[str] = []
    warnings: list[str] = []
    enabled = set(enabled_providers or [])

    for name in registry.list_registered():
        adapter = registry.get(name)
        if adapter is None:
            continue
        if hasattr(adapter, "check_readiness"):
            readiness = adapter.check_readiness()
        else:
            readiness = _generic_readiness(adapter, name, enabled)
        providers.append(readiness)
        if readiness.status == "available":
            available.append(readiness.provider_name)
        elif readiness.status == "degraded":
            degraded.append(readiness.provider_name)
            warnings.append(
                f"{readiness.provider_name}: {readiness.reason or readiness.reason_code or 'degraded'}"
            )
        elif readiness.status == "unavailable":
            unavailable.append(readiness.provider_name)
            warnings.append(
                f"{readiness.provider_name}: {readiness.reason or readiness.reason_code or 'unavailable'}"
            )

    if enabled:
        blocking = [
            provider
            for provider in providers
            if provider.provider_name in enabled and provider.status != "available"
        ]
    else:
        blocking = [provider for provider in providers if provider.status == "unavailable"]

    return ExternalProviderDoctorReport(
        providers=providers,
        available_providers=available,
        unavailable_providers=unavailable,
        degraded_providers=degraded,
        safe_to_run_external_enhanced=not blocking,
        warnings=warnings,
    )


def render_provider_doctor_text(report: ExternalProviderDoctorReport) -> str:
    """Human-readable text output for the provider doctor command."""

    lines = [
        "GovRAG External Provider Doctor",
        "",
    ]
    for readiness in report.providers:
        lines.append(f"{readiness.provider_name}: {readiness.status}")
        if readiness.reason_code:
            lines.append(f"  reason: {readiness.reason_code}")
        if readiness.reason:
            lines.append(f"  detail: {readiness.reason}")
        if readiness.install_hint:
            lines.append(f"  install: {readiness.install_hint}")
        if readiness.config_hint:
            lines.append(f"  config: {readiness.config_hint}")
        lines.append(f"  maturity: {readiness.integration_maturity}")
        lines.append(f"  runtime execution: {'yes' if readiness.runtime_execution_available else 'no'}")
        if readiness.runtime_execution_reason:
            lines.append(f"  runtime reason: {readiness.runtime_execution_reason}")
        if readiness.fallback_provider:
            lines.append(f"  fallback: {readiness.fallback_provider}")
            lines.append(
                f"  fallback visible: {'yes' if readiness.fallback_visible else 'no'}"
            )
        lines.append(
            f"  offline safe: {'yes' if readiness.safe_offline else 'no'}"
        )
        lines.append("")
    lines.append(
        f"safe_to_run_external_enhanced: {'yes' if report.safe_to_run_external_enhanced else 'no'}"
    )
    return "\n".join(lines).rstrip()


def _generic_readiness(
    adapter: Any,
    name: str,
    enabled: set[str],
) -> ProviderReadiness:
    if enabled and name not in enabled:
        return ProviderReadiness(
            provider_name=name,
            available=False,
            status="disabled",
            reason_code="disabled",
            reason="Provider is not enabled in the current configuration.",
            fallback_provider=_fallback_provider(name),
        )
    available = bool(adapter.is_available())
    return ProviderReadiness(
        provider_name=name,
        available=available,
        status="available" if available else "unavailable",
        reason_code=None if available else "package_missing",
        reason=None if available else f"Adapter '{name}' dependency is not available.",
        fallback_provider=_fallback_provider(name),
        integration_maturity="schema_only",
        runtime_execution_available=False,
    )


def _fallback_provider(name: str) -> str | None:
    mapping = {
        "refchecker_claim": "heuristic_claim_verifier",
        "refchecker_citation": "native_citation_verifier",
        "ragchecker": "native_context_metrics",
        "ragas": "native_retrieval_signals_only",
        "deepeval": "native_retrieval_signals_only",
        "cross_encoder_relevance": "lexical_overlap_relevance",
        "structured_llm_claim": "heuristic_claim_verifier",
        "structured_llm_citation": "native_citation_verifier",
    }
    return mapping.get(name)

