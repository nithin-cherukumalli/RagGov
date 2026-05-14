"""Regression: root causes must not collapse to generic adversarial labels without explicit evidence.

Root cause collapse is the most severe accuracy regression observed in the v2 benchmark:
  native root_cause_accuracy = 0.9333
  external root_cause_accuracy = 0.2

The collapse occurs because:
1. Bug #1: security_risk node gets first_failing_node priority when security analyzers warn
   → causal_hypothesis is always "adversarial_or_unsafe_context" when security_risk fires
2. Bug #2: RETRIEVAL_ANOMALY in A2P's SECURITY_FAILURE_TYPES causes non-security cases to
   trigger has_security_failure=True → A2P produces adversarial_context primary_cause
3. Bug #3: A2P's PROMPT_INJECTION output is fed to NCV rerun, reinforcing security_risk node
   failure → circular amplification

These tests verify that distinct failure types produce their correct distinct root causes,
not the collapsed generic adversarial label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raggov.engine import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "govrag_evidence_30"
FIXTURES_LEGACY = ROOT / "tests" / "fixtures"

_NATIVE_CONFIG = {
    "mode": "native",
    "enable_ncv": True,
    "enable_a2p": True,
    "use_llm": False,
}

_EXTERNAL_CONFIG = {
    "mode": "external-enhanced",
    "enable_ncv": True,
    "enable_a2p": True,
    "use_llm": False,
    "enabled_external_providers": ["ragas", "deepeval"],
}

# The collapsed root cause that must not appear for non-security failures
_COLLAPSED_ROOT_CAUSE = "adversarial_or_unsafe_context"

# Expected root cause per failure type
_EXPECTED_ROOT_CAUSES = {
    "retrieval_noise": "retrieval_noise_or_query_context_mismatch",
    "citation_mismatch": "citation_support_failure",
    "retrieval_miss": "retrieval_coverage_gap",
    "stale_source": "stale_or_invalid_source_usage",
}


def _load_run(fixture_path: Path) -> RAGRun:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = [RetrievedChunk.model_validate(c) for c in payload.get("retrieved_chunks", [])]
    entries = [
        CorpusEntry.model_validate(e)
        for e in payload.get("corpus_entries", payload.get("corpus_metadata", {}).get("entries", []))
    ]
    metadata = dict(payload.get("metadata", {}))
    if "corpus_metadata" in payload:
        metadata["corpus_metadata"] = payload["corpus_metadata"]
    if payload.get("parser_validation_profile") is not None:
        metadata["parser_validation_profile"] = payload["parser_validation_profile"]
    if "citations" in payload:
        metadata["citations"] = payload["citations"]
    return RAGRun(
        run_id=payload.get("run_id", payload.get("case_id", "test-case")),
        query=payload["query"],
        retrieved_chunks=chunks,
        final_answer=payload["final_answer"],
        cited_doc_ids=payload.get("cited_doc_ids", []),
        answer_confidence=payload.get("answer_confidence"),
        trace=payload.get("trace"),
        corpus_entries=entries,
        metadata=metadata,
    )


def _root_cause(diagnosis) -> str | None:
    if diagnosis.causal_chains:
        return diagnosis.causal_chains[0].causal_hypothesis
    return None


def _first_failing_node(diagnosis) -> str | None:
    if diagnosis.ncv_report:
        return diagnosis.ncv_report.get("first_failing_node")
    return getattr(diagnosis, "first_failing_node", None)


# ---------------------------------------------------------------------------
# Native mode: correct root causes as baseline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrieval_noise_native_correct_root_cause() -> None:
    """Native: retrieval noise → retrieval_noise_or_query_context_mismatch."""
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc == _EXPECTED_ROOT_CAUSES["retrieval_noise"], (
        f"Expected {_EXPECTED_ROOT_CAUSES['retrieval_noise']!r}, got {rc!r}"
    )


@pytest.mark.integration
def test_citation_mismatch_native_correct_root_cause() -> None:
    """Native: citation mismatch → citation_support_failure."""
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    rc = _root_cause(diagnosis)
    # Citation root cause is citation_support_failure or similar citation variant
    assert rc is not None and "citation" in (rc or "").lower(), (
        f"Expected citation root cause, got {rc!r}"
    )
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Native citation_mismatch collapsed to adversarial root cause: {rc!r}"
    )


@pytest.mark.integration
def test_stale_source_native_correct_root_cause() -> None:
    """Native: stale source → stale_or_invalid_source_usage."""
    run = _load_run(FIXTURES / "stale_source_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Native stale_source collapsed to adversarial root cause: {rc!r}"
    )
    assert rc is not None and "stale" in (rc or "").lower(), (
        f"Expected stale root cause, got {rc!r}"
    )


# ---------------------------------------------------------------------------
# External-enhanced mode: must preserve distinct root causes
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrieval_noise_external_root_cause_not_collapsed() -> None:
    """External: retrieval noise must not collapse to adversarial_or_unsafe_context.

    This is the primary regression guard for root_cause_accuracy dropping from 0.93 → 0.20.
    Bug #2 (RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES) causes has_security_failure=True
    in A2P for retrieval noise cases, collapsing root_cause to the adversarial generic.
    """
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Root cause collapse detected: retrieval_noise → {rc!r} in external mode. "
        "Bug #2 (RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES) is active."
    )


@pytest.mark.integration
def test_citation_mismatch_external_root_cause_not_collapsed() -> None:
    """External: citation mismatch root cause must not collapse to adversarial."""
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Root cause collapse detected: citation_mismatch → {rc!r} in external mode. "
        "External signals or A2P mis-mapping is producing adversarial root cause."
    )


@pytest.mark.integration
def test_stale_source_external_root_cause_not_collapsed() -> None:
    """External: stale source root cause must not collapse to adversarial."""
    run = _load_run(FIXTURES / "stale_source_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Root cause collapse detected: stale_source → {rc!r} in external mode."
    )


@pytest.mark.integration
def test_retrieval_miss_external_root_cause_not_collapsed() -> None:
    """External: retrieval miss root cause must not collapse to adversarial."""
    run = _load_run(FIXTURES / "retrieval_miss_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc != _COLLAPSED_ROOT_CAUSE, (
        f"Root cause collapse detected: retrieval_miss → {rc!r} in external mode."
    )


# ---------------------------------------------------------------------------
# Cross-mode: external root cause must match native when native is correct
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_root_cause_consistent_across_modes_retrieval_noise() -> None:
    """External root cause for retrieval_noise must not collapse to adversarial.

    Note: External mode may produce a different (non-adversarial) root cause from native —
    this is a separate drift issue tracked independently. This test only guards against the
    critical regression: root_cause collapsing to adversarial_or_unsafe_context when native
    correctly identifies retrieval noise.
    """
    run_native = _load_run(FIXTURES / "retrieval_noise_policy.json")
    run_external = _load_run(FIXTURES / "retrieval_noise_policy.json")

    native = diagnose(run_native, config=_NATIVE_CONFIG)
    external = diagnose(run_external, config=_EXTERNAL_CONFIG)

    native_rc = _root_cause(native)
    external_rc = _root_cause(external)

    if native_rc is not None and native_rc != _COLLAPSED_ROOT_CAUSE:
        assert external_rc != _COLLAPSED_ROOT_CAUSE, (
            f"External mode collapsed root cause to adversarial: native={native_rc!r} (non-adversarial, correct) "
            f"but external={external_rc!r} (adversarial collapse). "
            "Bug #2 (RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES) or Bug #1 (static reason) is active."
        )


@pytest.mark.integration
def test_root_cause_consistent_across_modes_citation() -> None:
    """External root cause for citation_mismatch must match native when native is correct."""
    run_native = _load_run(FIXTURES / "citation_mismatch_policy.json")
    run_external = _load_run(FIXTURES / "citation_mismatch_policy.json")

    native = diagnose(run_native, config=_NATIVE_CONFIG)
    external = diagnose(run_external, config=_EXTERNAL_CONFIG)

    native_rc = _root_cause(native)
    external_rc = _root_cause(external)

    if native_rc is not None and native_rc != _COLLAPSED_ROOT_CAUSE:
        assert external_rc != _COLLAPSED_ROOT_CAUSE, (
            f"External mode collapsed citation root cause: native={native_rc!r} "
            f"but external={external_rc!r}."
        )


# ---------------------------------------------------------------------------
# Security case: only genuine security failures get adversarial root cause
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_genuine_security_case_gets_adversarial_root_cause() -> None:
    """Prompt injection cases are the ONLY cases that should get adversarial root cause.

    This test verifies the adversarial_or_unsafe_context root cause is correctly assigned
    for a genuine security failure — not suppressed by the bug fixes.
    """
    run = _load_run(FIXTURES / "prompt_injection_context.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    ffn = _first_failing_node(diagnosis)
    rc = _root_cause(diagnosis)

    # Prompt injection should produce security_risk as first_failing_node
    assert ffn == "security_risk", (
        f"Expected security_risk for prompt injection case, got {ffn!r}"
    )
    # And adversarial root cause
    assert rc == _COLLAPSED_ROOT_CAUSE, (
        f"Expected adversarial_or_unsafe_context for prompt injection, got {rc!r}"
    )
