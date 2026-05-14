"""Tests for stresslab expectation evaluation."""

from __future__ import annotations

from raggov.models.diagnosis import Diagnosis, FailureStage, FailureType, SecurityRisk

from stresslab.cases import list_cases, load_case
from stresslab.evaluation import evaluate_case
from stresslab.mutations import FAILURE_INJECTION_ALIASES, normalize_failure_injection


def _diagnosis(
    *,
    primary_failure: FailureType,
    stage: FailureStage,
    should_have_answered: bool,
    secondary_failures: list[FailureType] | None = None,
) -> Diagnosis:
    return Diagnosis(
        run_id="run-1",
        primary_failure=primary_failure,
        secondary_failures=secondary_failures or [],
        root_cause_stage=stage,
        should_have_answered=should_have_answered,
        security_risk=SecurityRisk.NONE,
        recommended_fix="fix",
    )


def test_parse_hierarchy_case_matches_parsing_family() -> None:
    case = load_case("parse_hierarchy_loss_ms20")
    diagnosis = _diagnosis(
        primary_failure=FailureType.TABLE_STRUCTURE_LOSS,
        stage=FailureStage.PARSING,
        should_have_answered=True,
        secondary_failures=[FailureType.INSUFFICIENT_CONTEXT],
    )

    evaluation = evaluate_case(case, diagnosis)

    assert evaluation.matched_primary is True
    assert evaluation.matched_should_have_answered is True
    assert evaluation.matched_secondary is True
    assert evaluation.matched_overall is True


def test_abstention_case_matches_privacy_failure() -> None:
    case = load_case("abstention_required_private_fact")
    diagnosis = _diagnosis(
        primary_failure=FailureType.PRIVACY_VIOLATION,
        stage=FailureStage.SECURITY,
        should_have_answered=False,
        secondary_failures=[FailureType.POST_RATIONALIZED_CITATION],
    )

    evaluation = evaluate_case(case, diagnosis)

    assert evaluation.matched_primary is True
    assert evaluation.matched_should_have_answered is True
    assert evaluation.matched_overall is True


def test_all_curated_cases_use_supported_failure_injections() -> None:
    unsupported = []
    for case_id in list_cases():
        case = load_case(case_id)
        normalized = normalize_failure_injection(case.failure_injection)
        if normalized not in set(FAILURE_INJECTION_ALIASES.values()):
            unsupported.append((case_id, case.failure_injection, normalized))

    assert unsupported == []
