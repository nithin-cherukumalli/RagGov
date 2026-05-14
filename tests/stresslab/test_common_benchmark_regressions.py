from __future__ import annotations

from raggov.models.diagnosis import FailureStage, FailureType
from stresslab.cases.load import load_common_rag_failures
from stresslab.runners.rag_failure_runner import RAGFailureRunner


def _diagnose_common_case(case_id: str, *, mode: str = "native"):
    runner = RAGFailureRunner(mode=mode, suite="common")
    case = next(case for case in load_common_rag_failures() if case.case_id == case_id)
    return runner.engine.diagnose(runner._build_run(case))


def test_metadata_loss_does_not_return_clean() -> None:
    diagnosis = _diagnose_common_case("parser_metadata_missing_03")

    assert diagnosis.primary_failure == FailureType.METADATA_LOSS
    assert diagnosis.root_cause_stage == FailureStage.PARSING


def test_retrieval_miss_maps_to_retrieval_coverage() -> None:
    diagnosis = _diagnose_common_case("retrieval_miss_06")

    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT
    assert diagnosis.root_cause_stage == FailureStage.RETRIEVAL


def test_retrieval_noise_does_not_map_to_security() -> None:
    diagnosis = _diagnose_common_case("retrieval_noise_07")

    assert diagnosis.primary_failure == FailureType.RETRIEVAL_ANOMALY
    assert diagnosis.root_cause_stage == FailureStage.RETRIEVAL
    assert diagnosis.security_risk.value == "NONE"


def test_citation_related_but_non_supporting_maps_to_citation_support() -> None:
    diagnosis = _diagnose_common_case("citation_related_not_supporting_24")

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GROUNDING


def test_superseded_source_maps_to_version_validity() -> None:
    diagnosis = _diagnose_common_case("version_superseded_29")

    assert diagnosis.primary_failure == FailureType.STALE_RETRIEVAL
    assert diagnosis.root_cause_stage == FailureStage.RETRIEVAL


def test_unsupported_claim_maps_to_claim_support() -> None:
    diagnosis = _diagnose_common_case("retrieval_unsupported_answer_10")

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GROUNDING


def test_contradicted_claim_maps_to_claim_support() -> None:
    diagnosis = _diagnose_common_case("grounding_contradicted_18")

    assert diagnosis.primary_failure == FailureType.CONTRADICTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GROUNDING


def test_ambiguous_query_blocks_clean_or_requires_human_review() -> None:
    diagnosis = _diagnose_common_case("quality_ambiguous_query_40")

    assert diagnosis.primary_failure == FailureType.LOW_CONFIDENCE
    assert diagnosis.root_cause_stage == FailureStage.CONFIDENCE
    assert diagnosis.human_review_required
