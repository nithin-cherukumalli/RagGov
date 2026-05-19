from __future__ import annotations

import os

from scripts.smoke_groq_claim_verifier import run_smoke
from raggov.connectors.groq_client import build_groq_client_from_env


def test_build_groq_client_from_env_returns_unavailable_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client, reason = build_groq_client_from_env()

    assert client is None
    assert reason == "missing_api_key"


def test_smoke_returns_exit_code_2_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    report, exit_code = run_smoke(verbose=False)

    assert exit_code == 2
    assert report["connectivity"] == "skip"
    assert report["extraction"] == "skip"
    assert report["entailment"] == "skip"
    assert report["notes"]
    assert "GROQ_API_KEY" not in str(report)
