from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from raggov.cli import app
from stresslab.runners.launch_readiness import LaunchReadinessReport


runner = CliRunner()


def _report(*, passed: bool) -> LaunchReadinessReport:
    return LaunchReadinessReport(
        status="PASS" if passed else "FAIL",
        passed=passed,
        failure_reasons=[] if passed else ["false_clean_count=1. Inspect regressions."],
        warnings=[],
        gates={"false_clean_count == 0": passed},
        metrics={
            "false_clean_count": 0 if passed else 1,
            "false_incomplete_count": 0,
            "benchmark_accuracy": 1.0,
            "benchmark_accuracy_threshold": 0.95,
        },
        checks={},
    )


def test_launch_readiness_command_writes_reports_and_passes(monkeypatch) -> None:
    monkeypatch.setattr("raggov.cli.run_launch_readiness", lambda **kwargs: _report(passed=True))

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["launch-readiness"])

        assert result.exit_code == 0
        assert "PASS" in result.stdout
        payload = json.loads(Path("reports/launch_readiness_report.json").read_text())
        assert payload["status"] == "PASS"
        assert Path("reports/launch_readiness_report.md").exists()


def test_launch_readiness_command_fails_when_report_fails(monkeypatch) -> None:
    monkeypatch.setattr("raggov.cli.run_launch_readiness", lambda **kwargs: _report(passed=False))

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["launch-readiness"])

        assert result.exit_code == 1
        assert "FAIL" in result.stdout
