"""Evaluation helpers for diagnosis-native golden cases."""

from __future__ import annotations

from dataclasses import dataclass, field

from raggov.models.diagnosis import Diagnosis

from stresslab.cases import DiagnosisGoldenCase


@dataclass(frozen=True)
class DiagnosisGoldenEvaluation:
    """Evaluation summary comparing a diagnosis to a diagnosis-golden case."""

    case_id: str
    matched_primary: bool
    matched_stage: bool
    matched_should_have_answered: bool
    matched_secondary: bool
    matched_citation_faithfulness: bool
    matched_overall: bool
    notes: list[str] = field(default_factory=list)


def evaluate_diagnosis_case(
    case: DiagnosisGoldenCase,
    diagnosis: Diagnosis,
) -> DiagnosisGoldenEvaluation:
    """Evaluate a diagnosis against exact RagGov-native expectations."""
    matched_primary = diagnosis.primary_failure.value == case.expected_primary_failure
    matched_stage = diagnosis.root_cause_stage.value == case.expected_root_cause_stage
    matched_should_have_answered = (
        diagnosis.should_have_answered == case.expected_should_have_answered
    )

    expected_secondary = set(case.expected_secondary_failures)
    observed_secondary = {failure.value for failure in diagnosis.secondary_failures}
    matched_secondary = expected_secondary.issubset(observed_secondary)

    matched_citation_faithfulness = (
        case.expected_citation_faithfulness is None
        or diagnosis.citation_faithfulness == case.expected_citation_faithfulness
    )

    notes: list[str] = []
    if not matched_primary:
        notes.append(
            "primary mismatch: expected "
            f"{case.expected_primary_failure}, observed {diagnosis.primary_failure.value}"
        )
    if not matched_stage:
        notes.append(
            "stage mismatch: expected "
            f"{case.expected_root_cause_stage}, observed {diagnosis.root_cause_stage.value}"
        )
    if not matched_should_have_answered:
        notes.append(
            "should_have_answered mismatch: expected "
            f"{case.expected_should_have_answered}, observed {diagnosis.should_have_answered}"
        )
    if expected_secondary and not matched_secondary:
        notes.append(
            "secondary mismatch: expected subset "
            + ", ".join(sorted(expected_secondary))
            + "; observed "
            + ", ".join(sorted(observed_secondary))
        )
    if case.expected_citation_faithfulness is not None and not matched_citation_faithfulness:
        notes.append(
            "citation faithfulness mismatch: expected "
            f"{case.expected_citation_faithfulness}, observed {diagnosis.citation_faithfulness}"
        )

    return DiagnosisGoldenEvaluation(
        case_id=case.case_id,
        matched_primary=matched_primary,
        matched_stage=matched_stage,
        matched_should_have_answered=matched_should_have_answered,
        matched_secondary=matched_secondary,
        matched_citation_faithfulness=matched_citation_faithfulness,
        matched_overall=(
            matched_primary
            and matched_stage
            and matched_should_have_answered
            and matched_secondary
            and matched_citation_faithfulness
        ),
        notes=notes,
    )
