"""Integration tests for dry-run stresslab runners."""

from __future__ import annotations

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
from stresslab.runners import run_case, run_suite


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
    assert [summary.case_id for summary in result.results] == [
        "abstention_required_private_fact"
    ]
