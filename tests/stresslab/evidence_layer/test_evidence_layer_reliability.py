from __future__ import annotations

import pytest

from raggov.models.diagnosis import FailureType

from tests.stresslab.evidence_layer import (
    diagnose_fixture,
    diagnosis_signal_inventory,
    has_required_report,
    list_evidence_cases,
    load_evidence_case,
)


def _has_signal(diagnosis_signals: set[str], evidence: list[str], needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in str(item).lower() for item in diagnosis_signals) or any(
        lowered in str(item).lower() for item in evidence
    )


@pytest.mark.parametrize("case_name", list_evidence_cases())
def test_evidence_layer_cases_are_reliable_offline(case_name: str) -> None:
    case = load_evidence_case(case_name)

    diagnosis = diagnose_fixture(case)

    assert diagnosis is not None
    assert diagnosis.run_id == case.case_id
    assert diagnosis.degraded_external_mode == case.expected.expected_degraded_external_mode
    assert sorted(diagnosis.missing_external_providers) == sorted(
        case.expected.expected_missing_external_providers
    )

    for report_name in case.expected.required_reports:
        assert has_required_report(diagnosis, report_name), (
            f"{case.case_id}: missing required report {report_name}"
        )

    if case.expected.not_clean:
        assert diagnosis.primary_failure != FailureType.CLEAN, (
            f"{case.case_id}: expected non-clean diagnosis, got CLEAN"
        )
    else:
        assert diagnosis.primary_failure == FailureType.CLEAN, (
            f"{case.case_id}: expected CLEAN, got {diagnosis.primary_failure}"
        )

    signals = diagnosis_signal_inventory(diagnosis)
    for required_signal in case.expected.required_evidence_signals:
        assert _has_signal(signals, diagnosis.evidence, required_signal), (
            f"{case.case_id}: missing signal {required_signal!r}; "
            f"signals={sorted(signals)}"
        )


def test_prompt_injection_case_keeps_security_as_primary() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("prompt_injection_context"))

    assert diagnosis.primary_failure == FailureType.PROMPT_INJECTION
    assert diagnosis.citation_faithfulness != "post_rationalized"
    assert any("Detected" in evidence for evidence in diagnosis.evidence)


def test_no_claim_case_does_not_become_incomplete() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("no_claim_clean"))

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert "ClaimGroundingAnalyzer" in diagnosis.checks_skipped
    assert diagnosis.retrieval_diagnosis_report is not None
    assert diagnosis.retrieval_diagnosis_report.primary_failure_type.value in {
        "no_clear_retrieval_failure",
        "insufficient_evidence_to_diagnose",
    }


def test_missing_external_provider_case_stays_visible_without_crashing() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("missing_external_provider"))

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert diagnosis.degraded_external_mode is True
    assert "cross_encoder_relevance" in diagnosis.missing_external_providers
    assert diagnosis.retrieval_diagnosis_report is not None
    assert "external_retrieval_relevance_signal" in (
        diagnosis.retrieval_diagnosis_report.missing_reports
    )
