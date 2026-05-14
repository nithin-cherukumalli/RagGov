from __future__ import annotations

from scripts.evaluate_pinpointing import DEFAULT_GOLD_SET, _engine_config, _load_run, load_gold_set
from raggov.engine import diagnose


def _diagnosis(case_id: str):
    case = next(case for case in load_gold_set(DEFAULT_GOLD_SET).cases if case.case_id == case_id)
    return diagnose(_load_run(case.run_fixture), config=_engine_config(case.engine_config))


def test_retrieval_miss_maps_to_retrieval_coverage() -> None:
    diagnosis = _diagnosis("retrieval_miss")

    assert diagnosis.first_failing_node == "retrieval_coverage"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_coverage"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_coverage_gap"
    assert diagnosis.trust_decision is not None
    assert diagnosis.trust_decision.recommended_for_gating is False
    assert diagnosis.causal_chains[0].calibration_status == "uncalibrated"


def test_retrieval_noise_maps_to_retrieval_precision() -> None:
    diagnosis = _diagnosis("retrieval_noise")

    assert diagnosis.first_failing_node == "retrieval_precision"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_precision"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_noise_or_query_context_mismatch"


def test_citation_mismatch_maps_to_citation_support() -> None:
    diagnosis = _diagnosis("citation_mismatch")

    assert diagnosis.first_failing_node == "citation_support"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "citation_support"
    assert diagnosis.causal_chains[0].causal_hypothesis == "citation_support_failure"


def test_stale_source_maps_to_version_validity() -> None:
    diagnosis = _diagnosis("stale_source")

    assert diagnosis.first_failing_node == "version_validity"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "version_validity"
    assert diagnosis.causal_chains[0].causal_hypothesis == "stale_or_invalid_source_usage"


def test_non_clean_cases_with_pinpoint_emit_causal_chains() -> None:
    for case_id in (
        "retrieval_miss",
        "retrieval_noise",
        "citation_mismatch",
        "stale_source",
        "security_prompt_injection",
        "parser_damage",
        "unsupported_claim_with_context",
    ):
        diagnosis = _diagnosis(case_id)
        assert diagnosis.primary_failure.value != "CLEAN"
        assert diagnosis.pinpoint_findings
        assert diagnosis.causal_chains
        assert diagnosis.causal_chains[0].calibration_status == "uncalibrated"
        assert diagnosis.causal_chains[0].calibrated_confidence is None
        assert diagnosis.trust_decision is not None
        assert diagnosis.trust_decision.recommended_for_gating is False


def test_clean_pass_has_no_false_causal_chain_requirement() -> None:
    diagnosis = _diagnosis("clean_pass")

    assert diagnosis.primary_failure.value == "CLEAN"
    assert diagnosis.first_failing_node is None
