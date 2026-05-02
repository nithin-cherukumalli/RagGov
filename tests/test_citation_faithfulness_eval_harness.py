"""Tests for the citation faithfulness v0 evaluation harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_HARNESS_PATH = _SCRIPTS_DIR / "evaluate_citation_faithfulness.py"
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "citation_faithfulness_v0.jsonl"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "evaluate_citation_faithfulness", _HARNESS_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_citation_faithfulness"] = mod
    spec.loader.exec_module(mod)
    return mod


_harness = _load_harness()

load_fixture = _harness.load_fixture
build_run = _harness.build_run
evaluate_case = _harness.evaluate_case
compute_metrics = _harness.compute_metrics
report_to_dict = _harness.report_to_dict
main = _harness.main


def test_fixture_file_exists_and_loads_15_cases() -> None:
    cases = load_fixture(_FIXTURE_PATH)

    assert _FIXTURE_PATH.exists()
    assert len(cases) >= 15


def test_fixture_cases_have_required_keys() -> None:
    required = {
        "case_id",
        "description",
        "query",
        "retrieved_chunks",
        "generated_answer",
        "cited_doc_ids",
        "claim_evidence_records",
        "retrieval_evidence_profile",
        "expected_citation_faithfulness_report",
        "expected_analyzer_result",
    }

    for case in load_fixture(_FIXTURE_PATH):
        assert not required - case.keys(), case["case_id"]


def test_invalid_json_fails_clearly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        load_fixture(bad)


def test_build_run_attaches_claim_records_and_retrieval_profile() -> None:
    case = load_fixture(_FIXTURE_PATH)[0]
    run = build_run(case)

    assert run.final_answer == case["generated_answer"]
    assert run.metadata["claim_evidence_records"]
    assert run.retrieval_evidence_profile is not None


def test_harness_runs_on_small_fixture() -> None:
    case = load_fixture(_FIXTURE_PATH)[0]
    result = evaluate_case(case)

    assert result["case_id"] == case["case_id"]
    assert result["actual_status"] == case["expected_analyzer_result"]["status"]
    assert "actual_report" in result
    assert "expected_report" in result


def test_metrics_are_computed() -> None:
    results = [evaluate_case(c) for c in load_fixture(_FIXTURE_PATH)[:3]]
    report = compute_metrics(results)

    assert report.total_cases == 3
    assert 0.0 <= report.exact_match_accuracy <= 1.0
    assert report.unsupported_citation_precision >= 0.0
    assert report.phantom_citation_recall >= 0.0
    assert report.missing_citation_precision >= 0.0
    assert report.skip_count >= 0


def test_output_json_is_created_when_requested(tmp_path: Path) -> None:
    output = tmp_path / "citation_eval.json"
    main(fixture_path=_FIXTURE_PATH, output_path=output)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total_cases"] >= 15
    assert "exact_match_accuracy" in payload
    assert "case_mismatches" in payload


def test_full_fixture_regression_passes() -> None:
    results = [evaluate_case(c) for c in load_fixture(_FIXTURE_PATH)]
    report = compute_metrics(results)

    assert report.total_cases >= 15
    assert report.case_mismatches == []
    assert report.exact_match_accuracy == 1.0
    assert report.false_positive_count == 0
    assert report.false_negative_count == 0


def test_report_to_dict_has_required_metric_fields() -> None:
    results = [evaluate_case(c) for c in load_fixture(_FIXTURE_PATH)]
    payload = report_to_dict(compute_metrics(results))

    expected = {
        "exact_match_accuracy",
        "unsupported_citation_precision",
        "unsupported_citation_recall",
        "phantom_citation_precision",
        "phantom_citation_recall",
        "missing_citation_precision",
        "missing_citation_recall",
        "contradicted_citation_count",
        "false_positive_count",
        "false_negative_count",
        "skip_count",
    }
    assert not expected - payload.keys()
