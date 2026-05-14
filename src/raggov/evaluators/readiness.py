"""Structured readiness models for optional external providers."""

from __future__ import annotations

from importlib import metadata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReadinessStatus = Literal["available", "unavailable", "degraded", "disabled"]


class ProviderReadiness(BaseModel):
    """Fast, deterministic readiness result for one external provider."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    available: bool
    status: ReadinessStatus
    reason_code: str | None = None
    reason: str | None = None
    install_hint: str | None = None
    config_hint: str | None = None
    requires_network: bool = False
    requires_model_download: bool = False
    safe_offline: bool = True
    fallback_provider: str | None = None
    fallback_visible: bool = True
    adapter_version: str | None = None
    package_version: str | None = None
    integration_maturity: Literal[
        "schema_only",
        "mock_runner",
        "configured_runner",
        "native_library_runtime",
        "validated_runtime"
    ] = "schema_only"
    runtime_execution_available: bool = False
    runtime_execution_reason: str | None = None


class ExternalProviderDoctorReport(BaseModel):
    """Aggregate readiness report across registered providers."""

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderReadiness] = Field(default_factory=list)
    available_providers: list[str] = Field(default_factory=list)
    unavailable_providers: list[str] = Field(default_factory=list)
    degraded_providers: list[str] = Field(default_factory=list)
    safe_to_run_external_enhanced: bool = False
    warnings: list[str] = Field(default_factory=list)


def package_version_or_none(package_name: str) -> str | None:
    """Return installed package version when importlib.metadata can resolve it."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None

