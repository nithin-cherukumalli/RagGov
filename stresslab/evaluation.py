"""Semantic evaluation helpers for curated stresslab cases."""

from __future__ import annotations

from dataclasses import dataclass, field

from raggov.models.diagnosis import Diagnosis, FailureStage, FailureType

from stresslab.cases import StressCase


@dataclass(frozen=True)
class CaseEvaluation:
    """Evaluation summary comparing a diagnosis to a curated stress case."""

    case_id: str
    matched_primary: bool
    matched_should_have_answered: bool
    matched_secondary: bool
    matched_overall: bool
    expected_primary: str
    observed_primary: str
    observed_stage: str
    notes: list[str] = field(default_factory=list)


PRIMARY_EXPECTATIONS: dict[str, dict[str, set[str]]] = {
    "abstention_required": {
        "failure_types": {FailureType.PRIVACY_VIOLATION.value, FailureType.INSUFFICIENT_CONTEXT.value},
        "stages": {FailureStage.SECURITY.value, FailureStage.SUFFICIENCY.value},
    },
    "parse_hierarchy_loss": {
        "failure_types": {FailureType.HIERARCHY_FLATTENING.value, FailureType.TABLE_STRUCTURE_LOSS.value},
        "stages": {FailureStage.PARSING.value},
    },
    "parse_table_corruption": {
        "failure_types": {FailureType.TABLE_STRUCTURE_LOSS.value, FailureType.METADATA_LOSS.value},
        "stages": {FailureStage.PARSING.value},
    },
    "metadata_misread": {
        "failure_types": {FailureType.METADATA_LOSS.value, FailureType.TABLE_STRUCTURE_LOSS.value},
        "stages": {FailureStage.PARSING.value},
    },
    "retrieval_missing_critical_context": {
        "failure_types": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.SCOPE_VIOLATION.value},
        "stages": {FailureStage.RETRIEVAL.value, FailureStage.SUFFICIENCY.value},
    },
    "chunk_boundary_split": {
        "failure_types": {FailureType.INCONSISTENT_CHUNKS.value, FailureType.UNSUPPORTED_CLAIM.value},
        "stages": {FailureStage.CHUNKING.value, FailureStage.GROUNDING.value},
    },
    "cross_section_reasoning": {
        "failure_types": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.UNSUPPORTED_CLAIM.value},
        "stages": {FailureStage.RETRIEVAL.value, FailureStage.SUFFICIENCY.value, FailureStage.CHUNKING.value},
    },
    "embedding_semantic_drift": {
        "failure_types": {FailureType.RETRIEVAL_ANOMALY.value, FailureType.SCOPE_VIOLATION.value},
        "stages": {FailureStage.EMBEDDING.value, FailureStage.RETRIEVAL.value},
    },
    "embedding_structured_relationship": {
        "failure_types": {FailureType.SCOPE_VIOLATION.value, FailureType.INSUFFICIENT_CONTEXT.value},
        "stages": {FailureStage.RETRIEVAL.value, FailureStage.SUFFICIENCY.value},
    },
    "retrieval_ranking_instability": {
        "failure_types": {FailureType.RETRIEVAL_ANOMALY.value, FailureType.INSUFFICIENT_CONTEXT.value},
        "stages": {FailureStage.RETRIEVAL.value, FailureStage.EMBEDDING.value, FailureStage.SUFFICIENCY.value},
    },
    "oversegmentation": {
        "failure_types": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.UNSUPPORTED_CLAIM.value},
        "stages": {FailureStage.CHUNKING.value, FailureStage.SUFFICIENCY.value, FailureStage.GROUNDING.value},
    },
    "undersegmentation": {
        "failure_types": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.UNSUPPORTED_CLAIM.value},
        "stages": {FailureStage.CHUNKING.value, FailureStage.SUFFICIENCY.value, FailureStage.GROUNDING.value},
    },
}

SECONDARY_EXPECTATION_MAP: dict[str, set[str]] = {
    "privacy_leakage": {FailureType.PRIVACY_VIOLATION.value},
    "cross_section_reasoning": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.UNSUPPORTED_CLAIM.value},
    "retrieval_missing_critical_context": {FailureType.INSUFFICIENT_CONTEXT.value, FailureType.SCOPE_VIOLATION.value},
    "retrieval_ranking_instability": {FailureType.RETRIEVAL_ANOMALY.value, FailureType.INSUFFICIENT_CONTEXT.value},
    "embedding_semantic_drift": {FailureType.RETRIEVAL_ANOMALY.value},
    "chunk_boundary_split": {FailureType.INCONSISTENT_CHUNKS.value, FailureType.UNSUPPORTED_CLAIM.value},
}


def evaluate_case(case: StressCase, diagnosis: Diagnosis) -> CaseEvaluation:
    """Evaluate a diagnosis against a curated stress case using semantic expectations."""
    expected = PRIMARY_EXPECTATIONS.get(case.expected_primary_failure, {})
    expected_types = expected.get("failure_types", set())
    expected_stages = expected.get("stages", set())

    observed_primary = diagnosis.primary_failure.value
    observed_stage = diagnosis.root_cause_stage.value
    matched_primary = observed_primary in expected_types or observed_stage in expected_stages

    secondary_targets = set()
    for label in case.expected_secondary_failures:
        secondary_targets.update(SECONDARY_EXPECTATION_MAP.get(label, set()))
    observed_secondary = {failure.value for failure in diagnosis.secondary_failures}
    matched_secondary = (
        not secondary_targets
        or bool(observed_secondary & secondary_targets)
        or diagnosis.primary_failure.value in secondary_targets
    )

    matched_should_have_answered = (
        diagnosis.should_have_answered == case.expected_should_have_answered
    )

    notes: list[str] = []
    if not matched_primary:
        notes.append(
            f"primary mismatch: expected {case.expected_primary_failure}, "
            f"observed {observed_primary} at {observed_stage}"
        )
    if not matched_should_have_answered:
        notes.append(
            f"should_have_answered mismatch: expected {case.expected_should_have_answered}, "
            f"observed {diagnosis.should_have_answered}"
        )
    if secondary_targets and not matched_secondary:
        notes.append(
            "secondary mismatch: expected one of "
            + ", ".join(sorted(secondary_targets))
            + "; observed "
            + ", ".join(sorted(observed_secondary))
        )

    return CaseEvaluation(
        case_id=case.case_id,
        matched_primary=matched_primary,
        matched_should_have_answered=matched_should_have_answered,
        matched_secondary=matched_secondary,
        matched_overall=matched_primary and matched_should_have_answered and matched_secondary,
        expected_primary=case.expected_primary_failure,
        observed_primary=observed_primary,
        observed_stage=observed_stage,
        notes=notes,
    )
