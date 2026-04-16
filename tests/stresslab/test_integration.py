"""Integration tests for dry-run stresslab runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun
from stresslab.cases import load_case
from stresslab.evaluation import evaluate_case
from stresslab.runners import (
    render_suite_markdown,
    run_case,
    run_suite,
    write_suite_markdown_report,
    write_suite_report,
)


def test_run_case_returns_run_and_matching_diagnosis_in_dry_run() -> None:
    result = run_case("parse_hierarchy_loss_ms20", profile="lan", dry_run=True)

    assert result.case_id == "parse_hierarchy_loss_ms20"
    assert isinstance(result.run, RAGRun)
    assert isinstance(result.diagnosis, Diagnosis)
    assert result.diagnosis.run_id == result.run.run_id


def test_run_suite_returns_expected_case_summary_in_dry_run() -> None:
    result = run_suite(
        case_ids=["abstention_required_private_fact"],
        profile="lan",
        dry_run=True,
    )

    assert result.total_count == 1
    assert result.match_rate == 1.0
    assert [summary.case_id for summary in result.results] == [
        "abstention_required_private_fact"
    ]
    assert result.results[0].expectation_matched is True
    assert result.matched_count == 1
    assert result.observed_primary_failures == {"PRIVACY_VIOLATION": 1}
    assert result.observed_root_cause_stages == {"SECURITY": 1}


def test_all_curated_cases_execute_and_produce_structured_evaluations_in_dry_run() -> None:
    case_ids = [
        "abstention_required_private_fact",
        "chunk_boundary_split_ms11",
        "cross_section_reasoning_ms20",
        "embedding_semantic_drift_duplicates",
        "embedding_structured_relationship_ms9",
        "metadata_misread_ms1",
        "oversegmentation_ms15",
        "parse_hierarchy_loss_ms20",
        "parse_table_corruption_ms39",
        "retrieval_missing_critical_context_ms20",
        "retrieval_ranking_instability_duplicate_cluster",
        "undersegmentation_ms20",
    ]

    for case_id in case_ids:
        result = run_case(case_id, profile="lan", dry_run=True)
        evaluation = evaluate_case(load_case(case_id), result.diagnosis)

        assert result.case_id == case_id
        assert evaluation.case_id == case_id
        assert isinstance(evaluation.matched_overall, bool)
        assert result.run.metadata["normalized_failure_injection"]


def test_write_suite_report_persists_summary_artifact(tmp_path: Path) -> None:
    result = run_suite(
        case_ids=["abstention_required_private_fact", "parse_hierarchy_loss_ms20"],
        profile="lan",
        dry_run=True,
    )

    output_path = write_suite_report(result, tmp_path / "suite-report.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert payload["total_count"] == 2
    assert payload["matched_count"] <= payload["total_count"]
    assert "observed_primary_failures" in payload
    assert "observed_root_cause_stages" in payload
    assert "mismatched_case_ids" in payload
    assert "recurring_mismatch_notes" in payload
    assert len(payload["results"]) == 2


def test_render_and_write_suite_markdown_report(tmp_path: Path) -> None:
    result = run_suite(
        case_ids=["abstention_required_private_fact"],
        profile="lan",
        dry_run=True,
    )

    markdown = render_suite_markdown(result)
    output_path = write_suite_markdown_report(result, tmp_path / "suite-report.md")

    assert "# Stresslab Suite Report" in markdown
    assert "## Observed Primary Failures" in markdown
    assert "## Mismatches" in markdown
    assert output_path.read_text(encoding="utf-8") == markdown
