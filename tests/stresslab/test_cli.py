"""CLI tests for stresslab suite reporting."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import raggov.cli as cli_module
from raggov.cli import app
from stresslab.runners import DiagnosisGoldenSuiteResult


runner = CliRunner()


def test_stresslab_suite_command_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    json_path = tmp_path / "suite-report.json"
    md_path = tmp_path / "suite-report.md"

    result = runner.invoke(
        app,
        [
            "stresslab-suite",
            "--case-id",
            "abstention_required_private_fact",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_count"] == 1
    assert payload["matched_count"] == 1
    assert "# Stresslab Suite Report" in md_path.read_text(encoding="utf-8")


def test_stresslab_suite_command_enforces_min_match_rate(tmp_path: Path) -> None:
    json_path = tmp_path / "suite-report.json"
    md_path = tmp_path / "suite-report.md"

    result = runner.invoke(
        app,
        [
            "stresslab-suite",
            "--case-id",
            "abstention_required_private_fact",
            "--case-id",
            "chunk_boundary_split_ms11",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
            "--min-match-rate",
            "1.0",
        ],
    )

    assert result.exit_code != 0


def test_stresslab_diagnosis_command_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    json_path = tmp_path / "diagnosis-suite.json"
    md_path = tmp_path / "diagnosis-suite.md"

    result = runner.invoke(
        app,
        [
            "stresslab-diagnosis",
            "--case-id",
            "clean_pass",
            "--case-id",
            "parser_table_loss",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_count"] == 2
    assert payload["matched_count"] == 2
    assert "# Diagnosis Golden Suite Report" in md_path.read_text(encoding="utf-8")


def test_stresslab_diagnosis_command_enforces_min_match_rate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    json_path = tmp_path / "diagnosis-suite.json"
    md_path = tmp_path / "diagnosis-suite.md"

    def _fake_run_diagnosis_suite(case_ids: list[str]) -> DiagnosisGoldenSuiteResult:
        return DiagnosisGoldenSuiteResult(
            results=[],
            total_count=1,
            matched_count=0,
            match_rate=0.0,
            observed_primary_failures={},
            observed_root_cause_stages={},
            mismatched_case_ids=["clean_pass"],
            recurring_mismatch_notes=["forced mismatch"],
        )

    monkeypatch.setattr(cli_module, "run_diagnosis_suite", _fake_run_diagnosis_suite)

    result = runner.invoke(
        app,
        [
            "stresslab-diagnosis",
            "--case-id",
            "clean_pass",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
            "--min-match-rate",
            "1.0",
        ],
    )

    assert result.exit_code != 0
