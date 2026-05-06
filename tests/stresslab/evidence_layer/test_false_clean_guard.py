from __future__ import annotations

import pytest

from raggov.models.diagnosis import FailureType
from tests.stresslab.evidence_layer import diagnose_fixture, load_evidence_case


KNOWN_FAILURE_CASES = [
    "retrieval_miss_policy",
    "retrieval_noise_policy",
    "citation_mismatch_policy",
    "unsupported_claim_policy",
    "contradicted_claim_policy",
    "stale_source_policy",
    "parser_damage_table",
    "prompt_injection_context",
]


@pytest.mark.parametrize("case_name", KNOWN_FAILURE_CASES)
def test_known_failure_cases_never_return_clean(case_name: str) -> None:
    diagnosis = diagnose_fixture(load_evidence_case(case_name))

    assert diagnosis.primary_failure != FailureType.CLEAN, (
        f"{case_name}: false CLEAN with evidence={diagnosis.evidence[:6]}"
    )


def test_clean_case_stays_clean() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("clean_policy_native"))

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert diagnosis.primary_failure != FailureType.INCOMPLETE_DIAGNOSIS


def test_missing_external_provider_does_not_fake_full_external_success() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("missing_external_provider"))

    assert diagnosis.degraded_external_mode is True
    assert "cross_encoder_relevance" in diagnosis.missing_external_providers
    assert diagnosis.primary_failure == FailureType.CLEAN


def test_no_claim_clean_case_does_not_false_incomplete() -> None:
    diagnosis = diagnose_fixture(load_evidence_case("no_claim_clean"))

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert diagnosis.primary_failure != FailureType.INCOMPLETE_DIAGNOSIS
