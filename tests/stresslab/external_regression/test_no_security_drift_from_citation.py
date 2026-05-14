"""Regression: citation mismatch cases must not drift to security_risk in external-enhanced mode.

Citation mismatch is a citation-layer failure (citation_support node). External evaluators
(RAGAS faithfulness, RefChecker) may produce low-score signals, but these must remain advisory
and must not cause the NCV priority policy to override the correctly identified citation_support
first_failing_node.

Bugs being guarded:
- Bug #1: static reason string in _check_security_risk() always contains "poisoning", causing
  _is_explicit_security_failure() to return True for any security_risk warn/fail state.
- Bug #2: RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES means low retrieval scores from citation
  fixture can activate has_security_failure=True in A2P, producing adversarial root cause.
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

_CITATION_NODES = {"citation_support", "context_assembly"}
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


def _first_failing_node(diagnosis) -> str | None:
    if diagnosis.ncv_report:
        return diagnosis.ncv_report.get("first_failing_node")
    return getattr(diagnosis, "first_failing_node", None)


def _root_cause(diagnosis) -> str | None:
    if diagnosis.causal_chains:
        return diagnosis.causal_chains[0].causal_hypothesis
    return None


def _ncv_security_risk_status(diagnosis) -> str | None:
    if not diagnosis.ncv_report:
        return None
    for node in diagnosis.ncv_report.get("node_results", []):
        if node.get("node") == "security_risk":
            return node.get("status")
    return None


# ---------------------------------------------------------------------------
# Native mode baseline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_citation_mismatch_native_citation_node() -> None:
    """Native mode correctly identifies citation failure, not security failure."""
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    ffn = _first_failing_node(diagnosis)
    assert ffn in _CITATION_NODES, (
        f"Expected citation-layer node, got {ffn!r}"
    )
    assert ffn != _DISALLOWED_NODE, (
        "Regression in native mode: citation_mismatch classified as security_risk"
    )


@pytest.mark.integration
def test_citation_mismatch_native_root_cause_not_adversarial() -> None:
    """Native root cause for citation mismatch must not be adversarial."""
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc not in _DISALLOWED_ROOT_CAUSES, (
        f"Native mode produced adversarial root cause for citation_mismatch: {rc!r}"
    )


# ---------------------------------------------------------------------------
# External-enhanced mode: must not drift
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_citation_mismatch_external_stays_citation_node() -> None:
    """External-enhanced must not override citation_support with security_risk.

    RAGAS faithfulness or RefChecker citation signals are advisory evidence about
    citation quality, not security evidence. The NCV pipeline must not reclassify
    citation failures as adversarial/injection attacks.
    """
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    ffn = _first_failing_node(diagnosis)
    assert ffn != _DISALLOWED_NODE, (
        f"Regression: citation_mismatch drifted to security_risk in external mode. "
        f"first_failing_node={ffn!r}. "
        "External signals incorrectly overriding citation-domain diagnosis."
    )


@pytest.mark.integration
def test_citation_mismatch_external_root_cause_not_adversarial() -> None:
    """External mode must not collapse citation_mismatch root cause to adversarial.

    Bug #2 regression: if A2P sees has_security_failure=True (from RETRIEVAL_ANOMALY in
    SECURITY_FAILURE_TYPES) and an entailed claim, it produces adversarial_context as primary_cause,
    causing causal_hypothesis to become adversarial_or_unsafe_context even for citation failures.
    """
    run = _load_run(FIXTURES / "citation_mismatch_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    rc = _root_cause(diagnosis)
    assert rc not in _DISALLOWED_ROOT_CAUSES, (
        f"Regression: citation_mismatch root cause collapsed to {rc!r} in external mode. "
        "Bug #2 (RETRIEVAL_ANOMALY in SECURITY_FAILURE_TYPES) producing adversarial A2P output."
    )


@pytest.mark.integration
def test_citation_mismatch_external_not_worse_than_native() -> None:
    """External-enhanced must not degrade a correct native citation diagnosis."""
    run_native = _load_run(FIXTURES / "citation_mismatch_policy.json")
    run_external = _load_run(FIXTURES / "citation_mismatch_policy.json")

    native = diagnose(run_native, config=_NATIVE_CONFIG)
    external = diagnose(run_external, config=_EXTERNAL_CONFIG)

    native_ffn = _first_failing_node(native)
    external_ffn = _first_failing_node(external)

    if native_ffn in _CITATION_NODES:
        assert external_ffn not in (_DISALLOWED_NODE,), (
            f"External mode degraded citation diagnosis: native={native_ffn!r} (correct) "
            f"but external={external_ffn!r} (security_risk drift). "
            "External signals are overriding a correct native diagnosis."
        )
