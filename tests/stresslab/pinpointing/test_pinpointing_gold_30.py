from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.evaluate_pinpointing import (
    PinpointEvalExpected,
    evaluate_pinpointing,
    load_gold_set,
)


GOLD_V2_30 = Path("tests/fixtures/govrag_pinpoint_eval/pinpoint_eval_gold_v2_30.json")


def test_v2_30_gold_file_loads_and_has_30_cases() -> None:
    gold = load_gold_set(GOLD_V2_30)

    assert len(gold.cases) == 30


def test_every_case_has_required_expected_labels() -> None:
    gold = load_gold_set(GOLD_V2_30)

    for case in gold.cases:
        expected = case.expected
        assert isinstance(expected, PinpointEvalExpected)
        assert expected.primary_failure is not None
        assert hasattr(expected, "first_failing_node")
        assert hasattr(expected, "pinpoint_node")
        assert hasattr(expected, "root_cause")
        assert expected.fix_category is not None
        assert isinstance(expected.secondary_nodes, list)
        assert isinstance(expected.affected_claim_ids, list)
        assert isinstance(expected.affected_doc_ids, list)
        assert isinstance(expected.human_review_required, bool)
        assert isinstance(expected.recommended_for_gating, bool)
        assert expected.calibration_status == "uncalibrated"


def test_case_distribution_is_adequate() -> None:
    gold = load_gold_set(GOLD_V2_30)

    clean_count = sum(1 for case in gold.cases if case.expected.primary_failure == "CLEAN")
    primary_counts = Counter(case.expected.first_failing_node for case in gold.cases if case.expected.first_failing_node is not None)
    secondary_counts = Counter()
    for case in gold.cases:
        secondary_counts.update(case.expected.secondary_nodes)

    def total(node: str) -> int:
        return primary_counts[node] + secondary_counts[node]

    assert clean_count >= 3
    assert total("parser_validity") >= 3
    assert total("retrieval_coverage") >= 4
    assert total("retrieval_precision") >= 4
    assert total("citation_support") >= 4
    assert total("version_validity") >= 3
    assert total("claim_support") >= 4
    assert total("security_risk") >= 3
    assert total("context_assembly") >= 2
    assert total("answer_completeness") >= 2


def test_evaluation_script_runs_on_v2_30() -> None:
    summary = evaluate_pinpointing(GOLD_V2_30)

    assert summary.total_cases == 30
    assert 0.0 <= summary.primary_failure_accuracy <= 1.0
    assert 0.0 <= summary.first_failing_node_accuracy <= 1.0
    assert 0.0 <= summary.pinpoint_node_accuracy <= 1.0
    assert 0.0 <= summary.root_cause_accuracy <= 1.0
    assert summary.fix_category_accuracy is None or 0.0 <= summary.fix_category_accuracy <= 1.0
    assert summary.false_clean_count >= 0
    assert summary.false_incomplete_count >= 0
    assert summary.recommended_for_gating_true_count >= 0
    assert summary.calibrated_confidence_present_count >= 0
    assert summary.production_gating_decision_count >= 0


def test_honesty_counters_remain_clean() -> None:
    summary = evaluate_pinpointing(GOLD_V2_30)

    assert summary.recommended_for_gating_true_count == 0
    assert summary.calibrated_confidence_present_count == 0
    assert summary.production_gating_decision_count == 0
    assert summary.non_uncalibrated_count == 0


def test_false_clean_remains_zero() -> None:
    summary = evaluate_pinpointing(GOLD_V2_30)

    assert summary.false_clean_count == 0
