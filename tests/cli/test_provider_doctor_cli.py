from __future__ import annotations

import json

from typer.testing import CliRunner

from raggov.cli import app
from raggov.evaluators.readiness import ExternalProviderDoctorReport, ProviderReadiness


runner = CliRunner()


class _FakeRegistry:
    def readiness_report(self, enabled_providers=None):
        return ExternalProviderDoctorReport(
            providers=[
                ProviderReadiness(
                    provider_name="refchecker_claim",
                    available=False,
                    status="unavailable",
                    reason_code="spacy_model_missing",
                    reason="spaCy model en_core_web_sm is not installed.",
                    install_hint="python -m spacy download en_core_web_sm",
                    fallback_provider="heuristic_claim_verifier",
                ),
                ProviderReadiness(
                    provider_name="ragas",
                    available=True,
                    status="available",
                    fallback_provider="native_retrieval_signals_only",
                ),
            ],
            available_providers=["ragas"],
            unavailable_providers=["refchecker_claim"],
            degraded_providers=[],
            safe_to_run_external_enhanced=False,
            warnings=["refchecker_claim: spaCy model en_core_web_sm is not installed."],
        )


def test_provider_doctor_json_output_is_valid(monkeypatch) -> None:
    monkeypatch.setattr("raggov.cli.create_standard_registry", lambda config: _FakeRegistry())

    result = runner.invoke(app, ["providers", "doctor", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["providers"][0]["provider_name"] == "refchecker_claim"
    assert payload["providers"][0]["reason_code"] == "spacy_model_missing"


def test_provider_doctor_text_output_includes_install_hints(monkeypatch) -> None:
    monkeypatch.setattr("raggov.cli.create_standard_registry", lambda config: _FakeRegistry())

    result = runner.invoke(app, ["providers", "doctor", "--format", "text"])

    assert result.exit_code == 0
    assert "GovRAG External Provider Doctor" in result.stdout
    assert "python -m spacy download en_core_web_sm" in result.stdout
    assert "fallback: heuristic_claim_verifier" in result.stdout
