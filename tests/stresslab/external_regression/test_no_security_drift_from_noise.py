"""Regression: retrieval noise cases must not drift to security_risk in external-enhanced mode.

Bug #1: _check_security_risk() emits a static reason that always contains "poisoning",
causing _is_explicit_security_failure() to return True whenever any security analyzer warns.
This forces priority Rule 2 to override first_failing_node to security_risk regardless of
the actual failure type.

Bug #2: RETRIEVAL_ANOMALY is incorrectly included in SECURITY_FAILURE_TYPES in A2P, causing
retrieval noise cases to generate has_security_failure=True, producing adversarial_context
primary_cause which maps to security_risk.

These tests guard against both bugs by verifying that retrieval noise cases stay in the
retrieval_precision/retrieval_coverage domain in both native and external-enhanced modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raggov.engine import diagnose
from raggov.models.run import RAGRun
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "govrag_evidence_30"

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

_RETRIEVAL_NODES = {"retrieval_precision", "retrieval_coverage"}
_DISALLOWED_NODE = "security_risk"
_DISALLOWED_ROOT_CAUSES = {"adversarial_or_unsafe_context"}


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


def _ncv_security_risk_status(diagnosis) -> str | None:
    if not diagnosis.ncv_report:
        return None
    for node in diagnosis.ncv_report.get("node_results", []):
        if node.get("node") == "security_risk":
            return node.get("status")
    return None


def _first_failing_node(diagnosis) -> str | None:
    if diagnosis.ncv_report:
        return diagnosis.ncv_report.get("first_failing_node")
    return getattr(diagnosis, "first_failing_node", None)


def _root_cause(diagnosis) -> str | None:
    if diagnosis.causal_chains:
        return diagnosis.causal_chains[0].causal_hypothesis
    return None


# ---------------------------------------------------------------------------
# Native mode: sanity baseline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrieval_noise_native_stays_retrieval_precision() -> None:
    """Native mode correctly identifies retrieval_precision as first_failing_node."""
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    ffn = _first_failing_node(diagnosis)
    assert ffn in _RETRIEVAL_NODES, (
        f"Expected first_failing_node in {_RETRIEVAL_NODES}, got {ffn!r}"
    )
    assert ffn != _DISALLOWED_NODE, (
        "Bug #1 regression: retrieval_noise was classified as security_risk in native mode"
    )


@pytest.mark.integration
def test_retrieval_noise_native_root_cause_not_adversarial() -> None:
    """Native mode root cause must be retrieval_noise_or_query_context_mismatch, not adversarial."""
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc not in _DISALLOWED_ROOT_CAUSES, (
        f"Bug #2 regression: native mode produced adversarial root cause for retrieval noise: {rc!r}"
    )


@pytest.mark.integration
def test_retrieval_noise_native_security_node_not_fail() -> None:
    """Security_risk NCV node should not be FAIL for a retrieval noise case in native mode."""
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    sec_status = _ncv_security_risk_status(diagnosis)
    assert sec_status != "fail", (
        f"Bug #1 regression: security_risk node was FAIL for retrieval_noise in native mode: {sec_status!r}"
    )


# ---------------------------------------------------------------------------
# External-enhanced mode: must not drift
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrieval_noise_external_stays_retrieval_precision() -> None:
    """External-enhanced mode must not drift retrieval_noise to security_risk.

    Regression guard for Bug #1 (static reason string) + Bug #3 (A2P→NCV circular amplification).
    External signals (low RAGAS context precision) are advisory and must not override the
    native retrieval_precision classification.
    """
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    ffn = _first_failing_node(diagnosis)
    assert ffn != _DISALLOWED_NODE, (
        f"Regression detected: retrieval_noise drifted to security_risk in external-enhanced mode. "
        f"first_failing_node={ffn!r}. "
        "Bug #1 (static reason) or Bug #3 (A2P→NCV loop) is active."
    )
    assert ffn in _RETRIEVAL_NODES | {None}, (
        f"Expected retrieval_precision or retrieval_coverage, got {ffn!r}"
    )


@pytest.mark.integration
def test_retrieval_noise_external_root_cause_not_adversarial() -> None:
    """External-enhanced mode must not assign adversarial_or_unsafe_context to retrieval noise.

    Regression guard for Bug #2: RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES causes A2P to
    produce adversarial_context primary_cause for retrieval noise, collapsing root_cause to
    adversarial_or_unsafe_context.
    """
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc not in _DISALLOWED_ROOT_CAUSES, (
        f"Regression detected: root_cause collapsed to {rc!r} for retrieval_noise in external mode. "
        "Bug #2 (RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES) is active."
    )


@pytest.mark.integration
def test_retrieval_noise_external_security_node_not_fail() -> None:
    """Security_risk NCV node must not FAIL for retrieval_noise in external-enhanced mode.

    Regression guard for Bug #1: static reason string contains 'poisoning', causing
    _is_explicit_security_failure() to return True for all security_risk warn/fail states,
    forcing priority Rule 2 to override first_failing_node unconditionally.
    """
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    sec_status = _ncv_security_risk_status(diagnosis)
    assert sec_status != "fail", (
        f"Regression detected: security_risk node FAIL for retrieval_noise in external mode. "
        f"Status={sec_status!r}. Bug #1 (static reason) is active."
    )


# ---------------------------------------------------------------------------
# Mode consistency: external must not degrade
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrieval_noise_external_node_not_worse_than_native() -> None:
    """External-enhanced first_failing_node must not be farther from truth than native.

    If native correctly returns retrieval_precision but external returns security_risk,
    this is a direct regression — external mode is actively harmful.
    """
    run_native = _load_run(FIXTURES / "retrieval_noise_policy.json")
    run_external = _load_run(FIXTURES / "retrieval_noise_policy.json")

    native = diagnose(run_native, config=_NATIVE_CONFIG)
    external = diagnose(run_external, config=_EXTERNAL_CONFIG)

    native_ffn = _first_failing_node(native)
    external_ffn = _first_failing_node(external)

    if native_ffn in _RETRIEVAL_NODES:
        assert external_ffn not in (_DISALLOWED_NODE,), (
            f"External mode degraded: native={native_ffn!r} (correct) but "
            f"external={external_ffn!r} (security_risk drift). "
            "External signals are overriding a correct native diagnosis."
        )
