from __future__ import annotations

from scripts.evaluate_pinpointing import DEFAULT_GOLD_SET, _engine_config, _load_run, load_gold_set
from raggov.engine import diagnose
from raggov.models.chunk import RetrievedChunk


def _case(case_id: str):
    return next(case for case in load_gold_set(DEFAULT_GOLD_SET).cases if case.case_id == case_id)


def _run(case_id: str):
    case = _case(case_id)
    return _load_run(case.run_fixture), _engine_config(case.engine_config)


def _diagnose(case_id: str):
    run, config = _run(case_id)
    return diagnose(run, config=config)


def _diagnose_mutated(case_id: str, mutate):
    run, config = _run(case_id)
    mutated = run.model_copy(deep=True)
    mutate(mutated)
    return diagnose(mutated, config=config)


def _add_noise_chunk(run, *, chunk_id: str, source_doc_id: str, score: float = 0.09) -> None:
    run.retrieved_chunks.append(
        RetrievedChunk(
            chunk_id=chunk_id,
            source_doc_id=source_doc_id,
            text=(
                "This chunk discusses unrelated gardening tools, coastal weather, and "
                "festival logistics with no answer-bearing evidence."
            ),
            score=score,
        )
    )


def test_retrieval_miss_stays_retrieval_coverage_under_noisy_distractors() -> None:
    diagnosis = _diagnose_mutated(
        "retrieval_miss",
        lambda run: _add_noise_chunk(run, chunk_id="retrieval-miss-noise", source_doc_id="noise-doc-1"),
    )

    assert diagnosis.first_failing_node == "retrieval_coverage"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_coverage"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_coverage_gap"
    assert diagnosis.trust_decision is not None
    assert diagnosis.trust_decision.recommended_for_gating is False
    assert diagnosis.causal_chains[0].calibration_status == "uncalibrated"


def test_retrieval_noise_stays_retrieval_precision_without_missing_evidence_signal() -> None:
    diagnosis = _diagnose("retrieval_noise")

    assert diagnosis.first_failing_node == "retrieval_precision"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_precision"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_noise_or_query_context_mismatch"
    assert diagnosis.primary_failure.value != "CLEAN"


def test_generic_retrieval_anomaly_does_not_become_security_risk() -> None:
    def mutate(run) -> None:
        run.retrieved_chunks[0].score = 0.99
        if len(run.retrieved_chunks) > 1:
            run.retrieved_chunks[1].score = 0.18
        if len(run.retrieved_chunks) > 2:
            run.retrieved_chunks[2].score = 0.11

    diagnosis = _diagnose_mutated("retrieval_noise", mutate)

    assert diagnosis.primary_failure.value == "RETRIEVAL_ANOMALY"
    assert diagnosis.first_failing_node != "security_risk"
    assert diagnosis.first_failing_node == "retrieval_precision"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_noise_or_query_context_mismatch"


def test_prompt_injection_still_selects_security_risk() -> None:
    diagnosis = _diagnose("security_prompt_injection")

    assert diagnosis.first_failing_node == "security_risk"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "security_risk"
    assert diagnosis.causal_chains[0].causal_hypothesis == "adversarial_or_unsafe_context"
    assert diagnosis.trust_decision is not None
    assert diagnosis.trust_decision.recommended_for_gating is False


def test_citation_support_does_not_steal_retrieval_coverage() -> None:
    diagnosis = _diagnose("retrieval_miss")

    assert diagnosis.first_failing_node == "retrieval_coverage"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_coverage"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_coverage_gap"


def test_citation_mismatch_stays_citation_support_when_claim_otherwise_supported() -> None:
    diagnosis = _diagnose("citation_mismatch")

    assert diagnosis.first_failing_node == "citation_support"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "citation_support"
    assert diagnosis.causal_chains[0].causal_hypothesis == "citation_support_failure"


def test_version_validity_does_not_win_without_explicit_stale_evidence() -> None:
    diagnosis = _diagnose("citation_mismatch")

    assert diagnosis.first_failing_node != "version_validity"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "citation_support"


def test_stale_cited_source_beats_citation_support() -> None:
    diagnosis = _diagnose("stale_source")

    assert diagnosis.first_failing_node == "version_validity"
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "version_validity"
    assert diagnosis.causal_chains[0].causal_hypothesis == "stale_or_invalid_source_usage"


def test_clean_case_remains_clean_under_harmless_metadata_mutation() -> None:
    diagnosis = _diagnose_mutated(
        "clean_pass",
        lambda run: run.metadata.update({"harmless_debug_tag": "no-op", "extra_trace": {"note": "noop"}}),
    )

    assert diagnosis.primary_failure.value == "CLEAN"
    assert diagnosis.first_failing_node is None
    assert diagnosis.pinpoint_findings == []
    assert diagnosis.causal_chains == []


def test_failing_pinpoint_cases_emit_causal_chains_and_keep_honesty_flags() -> None:
    diagnoses = [
        _diagnose("retrieval_miss"),
        _diagnose("retrieval_noise"),
        _diagnose("citation_mismatch"),
        _diagnose("stale_source"),
        _diagnose("security_prompt_injection"),
        _diagnose("parser_damage"),
        _diagnose("unsupported_claim_with_context"),
        _diagnose_mutated(
            "retrieval_miss",
            lambda run: _add_noise_chunk(run, chunk_id="retrieval-miss-noise-2", source_doc_id="noise-doc-2"),
        ),
    ]

    for diagnosis in diagnoses:
        assert diagnosis.primary_failure.value != "CLEAN"
        assert diagnosis.pinpoint_findings
        assert diagnosis.causal_chains
        assert diagnosis.causal_chains[0].calibration_status == "uncalibrated"
        assert diagnosis.causal_chains[0].calibrated_confidence is None
        assert diagnosis.trust_decision is not None
        assert diagnosis.trust_decision.recommended_for_gating is False
