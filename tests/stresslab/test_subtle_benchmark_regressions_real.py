from __future__ import annotations

import pytest

from raggov.models.diagnosis import FailureStage, FailureType
from stresslab.cases.load import load_subtle_rag_failures
from stresslab.runners.rag_failure_runner import RAGFailureRunner


def _diagnose_subtle_case(case_id: str, *, mode: str = "native"):
    runner = RAGFailureRunner(mode=mode, suite="subtle")
    case = next(case for case in load_subtle_rag_failures() if case.case_id == case_id)
    return runner.engine.diagnose(runner._build_run(case))


@pytest.mark.parametrize(
    ("case_id", "expected_failure", "expected_stage"),
    [
        ("subtle_incomplete_answer_02", FailureType.INSUFFICIENT_CONTEXT, FailureStage.GENERATION),
        ("subtle_plausible_hallucination_03", FailureType.SCOPE_VIOLATION, FailureStage.RETRIEVAL),
        ("subtle_related_non_supporting_04", FailureType.CITATION_MISMATCH, FailureStage.GROUNDING),
        ("subtle_partial_support_05", FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
        ("subtle_answer_drift_06", FailureType.UNSUPPORTED_CLAIM, FailureStage.GENERATION),
        ("subtle_ambiguous_query_07", FailureType.LOW_CONFIDENCE, FailureStage.UNKNOWN),
        ("subtle_local_contradiction_08", FailureType.CONTRADICTED_CLAIM, FailureStage.GROUNDING),
        ("subtle_keyword_overlap_10", FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
        ("subtle_table_value_swap_11", FailureType.CONTRADICTED_CLAIM, FailureStage.GROUNDING),
        ("subtle_near_miss_retrieval_13", FailureType.INSUFFICIENT_CONTEXT, FailureStage.RETRIEVAL),
        ("subtle_constraint_override_14", FailureType.CONTRADICTED_CLAIM, FailureStage.GROUNDING),
        ("subtle_many_weak_citations_15", FailureType.CITATION_MISMATCH, FailureStage.GROUNDING),
    ],
)
def test_subtle_real_regressions_surface_expected_failure(
    case_id: str,
    expected_failure: FailureType,
    expected_stage: FailureStage,
) -> None:
    diagnosis = _diagnose_subtle_case(case_id)

    assert diagnosis.primary_failure == expected_failure
    assert diagnosis.root_cause_stage == expected_stage
    assert diagnosis.primary_failure != FailureType.CLEAN
    assert diagnosis.human_review_required()


def test_subtle_external_disagreement_creates_probe_and_blocks_clean() -> None:
    diagnosis = _diagnose_subtle_case("subtle_external_disagreement_09")

    assert diagnosis.primary_failure == FailureType.LOW_CONFIDENCE
    assert diagnosis.root_cause_stage == FailureStage.UNKNOWN
    assert diagnosis.external_diagnosis_probes
    assert any(probe.should_trigger_human_review for probe in diagnosis.external_diagnosis_probes)
    assert diagnosis.human_review_required()
