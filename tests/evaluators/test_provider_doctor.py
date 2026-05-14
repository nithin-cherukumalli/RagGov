from __future__ import annotations

from typing import Any

from raggov.engine import DiagnosisEngine
from raggov.evaluators.base import ExternalEvaluationResult, ExternalEvaluatorProvider
from raggov.evaluators.doctor import build_provider_doctor_report
from raggov.evaluators.readiness import (
    ExternalProviderDoctorReport,
    ProviderReadiness,
)
from raggov.models.run import RAGRun


class _FakeProvider:
    def __init__(self, readiness: ProviderReadiness) -> None:
        self.name = readiness.provider_name
        self._readiness = readiness

    def is_available(self) -> bool:
        return self._readiness.available

    def check_readiness(self) -> ProviderReadiness:
        return self._readiness


class _FakeRegistry:
    def __init__(self, providers: list[_FakeProvider]) -> None:
        self._providers = {provider.name: provider for provider in providers}

    def list_registered(self) -> list[str]:
        return list(self._providers.keys())

    def get(self, name_or_provider: str) -> Any | None:
        return self._providers.get(name_or_provider)

    def readiness_report(self, enabled_providers: list[str] | None = None) -> ExternalProviderDoctorReport:
        return build_provider_doctor_report(
            self,
            enabled_providers=enabled_providers,
        )

    def evaluate_enabled(self, run, enabled_providers, strict_mode=False):
        return []


def test_doctor_report_lists_available_and_unavailable_providers_correctly() -> None:
    registry = _FakeRegistry(
        [
            _FakeProvider(
                ProviderReadiness(
                    provider_name="ragas",
                    available=True,
                    status="available",
                    fallback_provider="native_retrieval_signals_only",
                )
            ),
            _FakeProvider(
                ProviderReadiness(
                    provider_name="refchecker_claim",
                    available=False,
                    status="unavailable",
                    reason_code="spacy_model_missing",
                    reason="spaCy model en_core_web_sm is not installed.",
                    install_hint="python -m spacy download en_core_web_sm",
                    fallback_provider="heuristic_claim_verifier",
                )
            ),
        ]
    )

    report = registry.readiness_report(enabled_providers=["ragas", "refchecker_claim"])

    assert "ragas" in report.available_providers
    assert "refchecker_claim" in report.unavailable_providers
    assert report.safe_to_run_external_enhanced is False


def test_diagnosis_metadata_includes_readiness_reason_codes(monkeypatch) -> None:
    fake_registry = _FakeRegistry(
        [
            _FakeProvider(
                ProviderReadiness(
                    provider_name="refchecker_claim",
                    available=False,
                    status="unavailable",
                    reason_code="spacy_model_missing",
                    reason="spaCy model en_core_web_sm is not installed.",
                    install_hint="python -m spacy download en_core_web_sm",
                    fallback_provider="heuristic_claim_verifier",
                )
            )
        ]
    )
    monkeypatch.setattr("raggov.engine.create_standard_registry", lambda config: fake_registry)
    fake_registry.evaluate_enabled = lambda run, enabled_providers, strict_mode=False: [
        ExternalEvaluationResult(
            provider=ExternalEvaluatorProvider.refchecker,
            adapter_name="refchecker_claim",
            succeeded=False,
            missing_dependency=True,
            error="refchecker_claim unavailable",
        )
    ]

    engine = DiagnosisEngine(
        {
            "mode": "external-enhanced",
            "enabled_external_providers": ["refchecker_claim"],
        },
        analyzers=[],
    )
    run = RAGRun(
        run_id="provider-readiness-test",
        query="q",
        final_answer="a",
        retrieved_chunks=[],
        metadata={},
    )
    diagnosis = engine.diagnose(run)

    assert diagnosis.degraded_external_mode is True
    assert diagnosis.missing_external_providers == ["refchecker_claim"]
    assert diagnosis.external_provider_readiness[0].reason_code == "spacy_model_missing"
    assert diagnosis.external_provider_readiness[0].install_hint == "python -m spacy download en_core_web_sm"
