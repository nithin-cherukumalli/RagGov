from __future__ import annotations

from pathlib import Path

from raggov import diagnose
from raggov.decision_policy import select_primary_failure_with_policy
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.taxonomy import FAILURE_PRIORITY


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _insufficient_context_result() -> AnalyzerResult:
    return AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="fail",
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        stage=FailureStage.SUFFICIENCY,
        evidence=[
            "[sufficiency:missing_temporal_or_freshness_requirement] missing temporal support"
        ],
    )


def _grounding_result(
    *,
    failure_type: FailureType,
    label: str,
    label_reason: str,
    candidate_chunk_ids: list[str] | None = None,
    neutral_chunk_ids: list[str] | None = None,
    contradicting_chunk_ids: list[str] | None = None,
    value_conflicts: list[dict[str, str]] | None = None,
    evidence_reason: str | None = None,
) -> AnalyzerResult:
    return AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=failure_type,
        stage=FailureStage.GROUNDING,
        evidence=["Claim grounding summary: total=1, entailed=0, unsupported=1, contradicted=0"],
        claim_results=[
            ClaimResult(
                claim_text="The answer contains an ungrounded factual claim.",
                label=label,  # type: ignore[arg-type]
                label_reason=label_reason,
                candidate_chunk_ids=candidate_chunk_ids or [],
                neutral_chunk_ids=neutral_chunk_ids or [],
                contradicting_chunk_ids=contradicting_chunk_ids or [],
                value_conflicts=value_conflicts or [],
                evidence_reason=evidence_reason,
            )
        ],
    )


def _select(results: list[AnalyzerResult]) -> FailureType:
    selected, _, _ = select_primary_failure_with_policy(results, {}, FAILURE_PRIORITY)
    return selected


def test_unsupported_no_evidence_does_not_count_as_candidate_backed_evidence() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                label="unsupported",
                label_reason="unsupported_no_evidence",
                candidate_chunk_ids=["chunk-1"],
                neutral_chunk_ids=["chunk-1"],
            ),
        ]
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT


def test_contradicted_no_evidence_does_not_count_as_candidate_backed_evidence() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.CONTRADICTED_CLAIM,
                label="contradicted",
                label_reason="contradicted_no_evidence",
                candidate_chunk_ids=["chunk-1"],
                contradicting_chunk_ids=["chunk-1"],
            ),
        ]
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT


def test_unsupported_claim_with_real_candidate_evidence_still_beats_generic_insufficient_context() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                label="unsupported",
                label_reason="unsupported_with_candidate_evidence",
                candidate_chunk_ids=["chunk-1"],
                neutral_chunk_ids=["chunk-1"],
            ),
        ]
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM


def test_insufficient_context_wins_when_all_unsupported_claims_have_no_evidence() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                label="unsupported",
                label_reason="unsupported_no_evidence",
                candidate_chunk_ids=["chunk-1", "chunk-2"],
                neutral_chunk_ids=["chunk-1", "chunk-2"],
            ),
        ]
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT


def test_default_explicit_contradiction_label_alone_does_not_bypass_guard() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.CONTRADICTED_CLAIM,
                label="contradicted",
                label_reason="explicit_contradiction",
                candidate_chunk_ids=["chunk-1"],
                contradicting_chunk_ids=["chunk-1"],
                evidence_reason="Contradictory evidence exists among top-k candidate chunks.",
            ),
        ]
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM


def test_value_conflict_counts_as_explicit_contradiction() -> None:
    selected = _select(
        [
            _insufficient_context_result(),
            _grounding_result(
                failure_type=FailureType.CONTRADICTED_CLAIM,
                label="contradicted",
                label_reason="value_conflict",
                candidate_chunk_ids=["chunk-1"],
                contradicting_chunk_ids=["chunk-1"],
                value_conflicts=[
                    {"claim_value": "12%", "evidence_value": "3%", "field": "increase"}
                ],
            ),
        ]
    )

    assert selected == FailureType.CONTRADICTED_CLAIM


def test_unsupported_claims_fixture_selects_unsupported_claim() -> None:
    run = RAGRun.model_validate_json((FIXTURES / "unsupported_claims.json").read_text())

    diagnosis = diagnose(run, config={"mode": "native"})

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM


def test_insufficient_context_fixture_remains_insufficient_context() -> None:
    run = RAGRun.model_validate_json((FIXTURES / "insufficient_context.json").read_text())

    diagnosis = diagnose(run, config={"mode": "native"})

    assert diagnosis.primary_failure == FailureType.INSUFFICIENT_CONTEXT


def test_clean_pass_fixture_remains_clean() -> None:
    run = RAGRun.model_validate_json((FIXTURES / "clean_pass.json").read_text())

    diagnosis = diagnose(run, config={"mode": "native"})

    assert diagnosis.primary_failure == FailureType.CLEAN
