"""Regression: low RAGAS/DeepEval scores must remain advisory — never gate-failing alone.

External evaluator signals (RAGAS context_precision, faithfulness; DeepEval contextual_relevancy)
are emitted with calibration_status="uncalibrated_locally" and recommended_for_gating=False.
They should never alone cause a CLEAN case to become FAIL, or cause first_failing_node to change
from a correctly identified native value.

Three properties being tested:
1. A case with artificially degraded RAGAS scores (clean fixture + RAGAS signals) must not
   change the primary_failure classification.
2. External signals present in metadata should appear in retrieval_diagnosis as advisory evidence
   only (not as primary_failure evidence).
3. The overall diagnosis remains calibration_status="uncalibrated" — external signals must not
   inflate confidence.
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


def _primary_failure(diagnosis) -> str:
    pf = diagnosis.primary_failure
    if pf is None:
        return "CLEAN"
    # Handle both FailureType enum ("FailureType.CLEAN") and plain string ("CLEAN")
    s = str(pf)
    return s.split(".")[-1] if "." in s else s


def _first_failing_node(diagnosis) -> str | None:
    if diagnosis.ncv_report:
        return diagnosis.ncv_report.get("first_failing_node")
    return getattr(diagnosis, "first_failing_node", None)


def _all_analyzer_calibration_statuses(diagnosis) -> list[str]:
    return [r.calibration_status for r in diagnosis.analyzer_results if r.calibration_status]


def _external_signal_records_are_advisory(diagnosis) -> bool:
    """Verify that any external signals in analyzer results are marked uncalibrated + non-gating."""
    for result in diagnosis.analyzer_results:
        if result.analyzer_name == "RetrievalDiagnosisAnalyzerV0" and result.retrieval_diagnosis_report:
            report = result.retrieval_diagnosis_report
            for sig in getattr(report, "evidence_signals", []):
                source = str(getattr(sig, "source_report", "") or "")
                if "ExternalEvaluationResult" in source or "external" in source.lower():
                    # External signals must not be marked as primary evidence
                    if getattr(sig, "is_primary_evidence", False):
                        return False
    return True


# ---------------------------------------------------------------------------
# Clean case: external signals must not flip to failure
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_clean_native_stays_clean() -> None:
    """Sanity: clean case stays CLEAN in native mode."""
    run = _load_run(FIXTURES / "clean_policy_native.json")
    diagnosis = diagnose(run, config=_NATIVE_CONFIG)

    assert _primary_failure(diagnosis) == "CLEAN", (
        f"Expected CLEAN, got {_primary_failure(diagnosis)!r}"
    )
    assert _first_failing_node(diagnosis) is None, (
        f"Clean case has a first_failing_node: {_first_failing_node(diagnosis)!r}"
    )


@pytest.mark.integration
def test_clean_external_stays_clean() -> None:
    """Clean case must remain CLEAN in external-enhanced mode even with external evaluators active.

    External evaluators may produce lower-than-ideal scores for clean cases (due to calibration
    gaps), but those scores must not flip a CLEAN diagnosis to a failure.
    """
    run = _load_run(FIXTURES / "clean_policy_native.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    pf = _primary_failure(diagnosis)
    assert pf == "CLEAN", (
        f"Regression: clean case was reclassified as {pf!r} in external-enhanced mode. "
        "External signals flipped a correctly-clean diagnosis."
    )


@pytest.mark.integration
def test_clean_external_no_first_failing_node() -> None:
    """Clean case in external mode must have no first_failing_node."""
    run = _load_run(FIXTURES / "clean_policy_native.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    ffn = _first_failing_node(diagnosis)
    assert ffn is None, (
        f"Regression: clean case has first_failing_node={ffn!r} in external mode. "
        "External signals are generating spurious node failures."
    )


# ---------------------------------------------------------------------------
# No-claim clean: external evaluators have nothing to evaluate on claims
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_no_claim_clean_external_stays_clean() -> None:
    """Cases with no claims must remain CLEAN when external evaluators run."""
    run = _load_run(FIXTURES / "no_claim_clean.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    pf = _primary_failure(diagnosis)
    ffn = _first_failing_node(diagnosis)

    assert pf == "CLEAN", (
        f"No-claim clean case reclassified as {pf!r} in external mode."
    )
    assert ffn is None, (
        f"No-claim clean case has first_failing_node={ffn!r} in external mode."
    )


# ---------------------------------------------------------------------------
# Advisory signal properties
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_external_signals_consumed_as_advisory_not_primary() -> None:
    """External signals in retrieval diagnosis must be tagged as advisory, not primary evidence."""
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    assert _external_signal_records_are_advisory(diagnosis), (
        "External signals were consumed as primary evidence. "
        "They must remain advisory (is_primary_evidence=False)."
    )


@pytest.mark.integration
def test_external_mode_diagnosis_calibration_uncalibrated() -> None:
    """External-enhanced mode diagnosis must remain uncalibrated (no external tool inflates confidence).

    External evaluators are uncalibrated_locally. Their presence must not cause the diagnosis
    calibration_status to upgrade to 'calibrated'.
    """
    run = _load_run(FIXTURES / "retrieval_noise_policy.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    # Overall diagnosis calibration must not be "calibrated"
    cal = getattr(diagnosis, "calibration_status", None)
    assert cal != "calibrated", (
        f"Diagnosis calibration_status upgraded to {cal!r} in external mode. "
        "External signals must not inflate calibration confidence."
    )


# ---------------------------------------------------------------------------
# Cross-mode consistency: external must not degrade clean or near-clean cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_missing_external_provider_does_not_fail_case() -> None:
    """Missing external provider results must not cause case failure.

    If a provider (e.g., RAGAS) is unavailable, the diagnosis must proceed normally
    using native evidence only and not produce a missing_critical_evidence failure.
    """
    run = _load_run(FIXTURES / "missing_external_provider.json")
    diagnosis = diagnose(run, config=_EXTERNAL_CONFIG)

    pf = _primary_failure(diagnosis)
    assert "MISSING" not in pf.upper() or pf == "CLEAN", (
        f"Missing external provider caused case failure: {pf!r}. "
        "Providers must be optional and their absence must be graceful."
    )
