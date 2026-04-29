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


def test_stresslab_claim_diagnosis_command_writes_json_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim-diagnosis"
    gold_path = Path("stresslab/cases/golden/claim_diagnosis_gold_v0.json")

    result = runner.invoke(
        app,
        [
            "stresslab-claim-diagnosis",
            "--gold-set",
            str(gold_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    json_path = output_dir / "claim_diagnosis_report.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["evaluation_status"] == "diagnostic_gold_v0_small_unvalidated"
    assert payload["a2p_mode"] == "v1"
    assert "aggregate_metrics" in payload
    assert "claim_label_accuracy" in payload["aggregate_metrics"]


def test_stresslab_claim_diagnosis_command_writes_markdown_with_mismatch_section(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim-diagnosis"
    gold_path = Path("stresslab/cases/golden/claim_diagnosis_gold_v0.json")

    result = runner.invoke(
        app,
        [
            "stresslab-claim-diagnosis",
            "--gold-set",
            str(gold_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "markdown",
        ],
    )

    assert result.exit_code == 0
    md_path = output_dir / "claim_diagnosis_report.md"
    assert md_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Claim-Level Diagnostic Evaluation Report" in markdown
    assert "## Mismatches" in markdown


def test_stresslab_claim_diagnosis_command_defaults_to_v1_gold_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim-diagnosis"

    result = runner.invoke(
        app,
        [
            "stresslab-claim-diagnosis",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads((output_dir / "claim_diagnosis_report.json").read_text(encoding="utf-8"))
    assert payload["evaluation_status"] == "diagnostic_gold_v1_large_unvalidated"
    assert payload["a2p_mode"] == "v1"
    assert payload["case_count"] >= 50
    assert "category_metrics" in payload


def test_stresslab_claim_diagnosis_command_accepts_enable_a2p_v2(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim-diagnosis"
    gold_path = Path("stresslab/cases/golden/claim_diagnosis_gold_v0.json")

    result = runner.invoke(
        app,
        [
            "stresslab-claim-diagnosis",
            "--gold-set",
            str(gold_path),
            "--output-dir",
            str(output_dir),
            "--enable-a2p-v2",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads((output_dir / "claim_diagnosis_report.json").read_text(encoding="utf-8"))
    assert payload["a2p_mode"] == "v2"


def test_stresslab_claim_diagnosis_command_rejects_no_enable_a2p_with_enable_a2p_v2(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim-diagnosis"
    gold_path = Path("stresslab/cases/golden/claim_diagnosis_gold_v0.json")

    result = runner.invoke(
        app,
        [
            "stresslab-claim-diagnosis",
            "--gold-set",
            str(gold_path),
            "--output-dir",
            str(output_dir),
            "--no-enable-a2p",
            "--enable-a2p-v2",
            "--format",
            "json",
        ],
    )

    assert result.exit_code != 0
    assert "--enable-a2p-v2 requires A2P to be enabled" in result.stdout
