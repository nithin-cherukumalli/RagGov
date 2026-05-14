from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from raggov.evaluators.citation.refchecker_adapter import RefCheckerCitationSignalProvider
from raggov.evaluators.claim.refchecker_adapter import RefCheckerClaimSignalProvider
from raggov.evaluators.retrieval.cross_encoder import CrossEncoderRetrievalRelevanceProvider
from raggov.evaluators.retrieval.ragchecker_adapter import RAGCheckerSignalProvider


def test_refchecker_claim_package_missing_returns_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider().check_readiness()

    assert readiness.available is False
    assert readiness.reason_code == "package_missing"
    assert readiness.install_hint == "pip install refchecker"

def test_refchecker_claim_importable_but_no_runner_returns_schema_only(monkeypatch: pytest.MonkeyPatch) -> None:
    refchecker_module = SimpleNamespace()
    
    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        if name == "spacy":
            return SimpleNamespace()
        if name == "en_core_web_sm":
            return SimpleNamespace()
        return SimpleNamespace()
        
    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider().check_readiness()

    assert readiness.available is False
    assert readiness.status == "degraded"
    assert readiness.reason_code == "runtime_execution_not_configured"
    assert readiness.integration_maturity == "schema_only"

def test_refchecker_claim_with_metric_results_returns_mock_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    refchecker_module = SimpleNamespace()
    
    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        if name == "spacy":
            return SimpleNamespace()
        if name == "en_core_web_sm":
            return SimpleNamespace()
        return SimpleNamespace()
        
    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider({"metric_results": {}}).check_readiness()

    assert readiness.available is True
    assert readiness.status == "available"
    assert readiness.integration_maturity == "mock_runner"

def test_refchecker_claim_with_claim_runner_returns_configured_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    refchecker_module = SimpleNamespace()
    
    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        if name == "spacy":
            return SimpleNamespace()
        if name == "en_core_web_sm":
            return SimpleNamespace()
        return SimpleNamespace()
        
    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider({"claim_runner": lambda x: x}).check_readiness()

    assert readiness.available is True
    assert readiness.status == "available"
    assert readiness.integration_maturity == "configured_runner"



def test_refchecker_claim_spacy_missing_returns_spacy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    refchecker_module = SimpleNamespace()

    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        if name == "spacy":
            raise ImportError(name)
        return SimpleNamespace()

    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider().check_readiness()

    assert readiness.available is False
    assert readiness.reason_code == "spacy_missing"


def test_refchecker_claim_spacy_model_missing_returns_spacy_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    refchecker_module = SimpleNamespace()

    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        if name == "spacy":
            return SimpleNamespace()
        if name == "en_core_web_sm":
            raise ImportError(name)
        return SimpleNamespace()

    monkeypatch.setattr("raggov.evaluators.claim.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerClaimSignalProvider().check_readiness()

    assert readiness.available is False
    assert readiness.reason_code == "spacy_model_missing"
    assert "en_core_web_sm" in (readiness.install_hint or "")


def test_refchecker_citation_adapter_api_mismatch_returns_adapter_api_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refchecker_module = SimpleNamespace()

    def fake_import(name: str):
        if name == "refchecker":
            return refchecker_module
        return SimpleNamespace()

    monkeypatch.setattr("raggov.evaluators.citation.refchecker_adapter.importlib.import_module", fake_import)
    readiness = RefCheckerCitationSignalProvider(
        {"refchecker_required_attr": "Checker"}
    ).check_readiness()

    assert readiness.available is False
    assert readiness.reason_code == "adapter_api_mismatch"


def test_provider_disabled_returns_disabled() -> None:
    readiness = RefCheckerClaimSignalProvider(
        {"enabled_external_providers": ["ragas"]}
    ).check_readiness()

    assert readiness.status == "disabled"
    assert readiness.reason_code == "disabled"


def test_cross_encoder_offline_unsafe_provider_is_marked_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raggov.evaluators.retrieval.cross_encoder.importlib.util.find_spec",
        lambda name: object() if name == "sentence_transformers" else None,
    )
    monkeypatch.setattr(
        "raggov.evaluators.retrieval.cross_encoder._cross_encoder_model_cached",
        lambda model_name: False,
    )
    readiness = CrossEncoderRetrievalRelevanceProvider(
        {"enabled_external_providers": ["cross_encoder_relevance"]}
    ).check_readiness()

    assert readiness.available is False
    assert readiness.status == "degraded"
    assert readiness.reason_code == "offline_resource_missing"
    assert readiness.safe_offline is False


def test_ragchecker_import_path_mismatch_returns_import_path_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("raggov.evaluators.retrieval.ragchecker_adapter.importlib.import_module", fake_import)
    readiness = RAGCheckerSignalProvider(
        {"ragchecker_import_name": "ragchecker.custom"}
    ).check_readiness()

    assert readiness.available is False
    assert readiness.reason_code == "import_path_mismatch"

def test_ragchecker_importable_but_no_runner_returns_schema_only(monkeypatch: pytest.MonkeyPatch) -> None:
    ragchecker_module = SimpleNamespace()

    def fake_import(name: str):
        if name == "ragchecker":
            return ragchecker_module
        raise ImportError(name)

    monkeypatch.setattr("raggov.evaluators.retrieval.ragchecker_adapter.importlib.import_module", fake_import)
    monkeypatch.setattr("raggov.evaluators.retrieval.ragchecker_adapter.package_version_or_none", lambda x: "1.0")
    
    # We must patch find_spec since it calls it
    monkeypatch.setattr("raggov.evaluators.retrieval.ragchecker_adapter.importlib.util.find_spec", lambda x: True)

    readiness = RAGCheckerSignalProvider().check_readiness()

    assert readiness.available is False
    assert readiness.status == "degraded"
    assert readiness.reason_code == "runtime_execution_not_configured"
    assert readiness.integration_maturity == "schema_only"
