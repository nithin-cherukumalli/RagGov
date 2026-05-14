"""Tests for diagnosis-native golden evaluation."""

from __future__ import annotations

from raggov.models.diagnosis import Diagnosis, FailureStage, FailureType, SecurityRisk
from stresslab.cases import DiagnosisGoldenCase
from stresslab.diagnosis_evaluation import evaluate_diagnosis_case


def test_evaluate_diagnosis_case_matches_exact_expectations() -> None:
    case = DiagnosisGoldenCase(
        case_id="gold-case",
        run_fixture="fixtures/clean_pass.json",
        expected_primary_failure="CLEAN",
        expected_root_cause_stage="UNKNOWN",
        expected_should_have_answered=True,
        expected_secondary_failures=["POST_RATIONALIZED_CITATION"],
        expected_citation_faithfulness="genuine",
    )
    diagnosis = Diagnosis(
        run_id="run-1",
        primary_failure=FailureType.CLEAN,
        secondary_failures=[FailureType.POST_RATIONALIZED_CITATION],
        root_cause_stage=FailureStage.UNKNOWN,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        recommended_fix="None",
        citation_faithfulness="genuine",
    )

    evaluation = evaluate_diagnosis_case(case, diagnosis)

    assert evaluation.matched_primary is True
    assert evaluation.matched_stage is True
    assert evaluation.matched_secondary is True
    assert evaluation.matched_citation_faithfulness is True
    assert evaluation.matched_overall is True


def test_evaluate_diagnosis_case_reports_mismatches() -> None:
    case = DiagnosisGoldenCase(
        case_id="gold-case",
        run_fixture="fixtures/clean_pass.json",
        expected_primary_failure="UNSUPPORTED_CLAIM",
        expected_root_cause_stage="GROUNDING",
        expected_should_have_answered=False,
        expected_secondary_failures=["STALE_RETRIEVAL"],
        expected_citation_faithfulness="post_rationalized",
    )
    diagnosis = Diagnosis(
        run_id="run-1",
        primary_failure=FailureType.CLEAN,
        secondary_failures=[],
        root_cause_stage=FailureStage.UNKNOWN,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        recommended_fix="None",
        citation_faithfulness="genuine",
    )

    evaluation = evaluate_diagnosis_case(case, diagnosis)

    assert evaluation.matched_overall is False
    assert len(evaluation.notes) == 5
