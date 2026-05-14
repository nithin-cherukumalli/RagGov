from __future__ import annotations

from pathlib import Path

from scripts.evaluate_pinpointing import (
    DEFAULT_GOLD_SET,
    PinpointEvalSummary,
    determine_exit_code,
    evaluate_pinpointing,
    load_gold_set,
    main,
    render_markdown,
)


def test_gold_file_loads() -> None:
    gold = load_gold_set(DEFAULT_GOLD_SET)

    assert gold.evaluation_status
    assert gold.cases
    assert all(case.case_id for case in gold.cases)
    assert all(case.run_fixture for case in gold.cases)
    assert all(case.expected.primary_failure for case in gold.cases)


def test_evaluation_script_runs_on_small_gold_set() -> None:
    summary = evaluate_pinpointing(DEFAULT_GOLD_SET)

    assert isinstance(summary, PinpointEvalSummary)
    assert summary.total_cases > 0
    assert 0.0 <= summary.first_failing_node_accuracy <= 1.0
    assert 0.0 <= summary.root_cause_accuracy <= 1.0
    assert summary.false_clean_count >= 0
    assert summary.recommended_for_gating_true_count >= 0


def test_markdown_output_works() -> None:
    summary = evaluate_pinpointing(DEFAULT_GOLD_SET)
    markdown = render_markdown(summary)

    assert "| metric | value |" in markdown
    assert "first_failing_node_accuracy" in markdown
    assert "root_cause_accuracy" in markdown


def test_trace_case_and_markdown_trace_output_work() -> None:
    summary = evaluate_pinpointing(DEFAULT_GOLD_SET, dump_traces=True, trace_case="retrieval_miss")
    markdown = render_markdown(summary)

    assert summary.total_cases == 1
    assert summary.per_case[0].case_id == "retrieval_miss"
    assert summary.per_case[0].trace is not None
    assert "## Traces" in markdown
    assert "\"case_id\": \"retrieval_miss\"" in markdown


def test_fail_under_node_accuracy_exits_when_threshold_too_high() -> None:
    summary = evaluate_pinpointing(DEFAULT_GOLD_SET)

    assert determine_exit_code(summary, fail_under_node_accuracy=1.1) == 1


def test_gating_honesty_check_counts_and_fails() -> None:
    baseline = evaluate_pinpointing(DEFAULT_GOLD_SET)
    violating = PinpointEvalSummary(
        **{
            **baseline.__dict__,
            "recommended_for_gating_true_count": 1,
        }
    )

    assert violating.recommended_for_gating_true_count == 1
    assert determine_exit_code(violating, fail_on_gating_violations=True) == 1


def test_safe_no_network_no_llm_behavior(capsys) -> None:
    exit_code = main(["--gold-set", str(DEFAULT_GOLD_SET), "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "\"total_cases\"" in captured.out
