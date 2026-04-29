"""Tests for the Day 1 baseline freeze runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

freeze_module = importlib.import_module("stresslab.runners.freeze_day1_baseline")
from stresslab.runners.freeze_day1_baseline import (
    CLAIM_REPORT_JSON_NAME,
    CLAIM_REPORT_MD_NAME,
    TEST_REPORT_NAME,
    freeze_day1_baseline,
)


def test_freeze_day1_baseline_writes_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    def _fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        assert cwd == repo_root

        if command == ["pytest", "-q"]:
            return CompletedProcess(command, 0, stdout="...\n339 passed in 38.54s\n", stderr="")

        assert command[:4] == ["raggov", "stresslab-claim-diagnosis", "--format", "both"]
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "claim_diagnosis_report.json").write_text(
            json.dumps(
                {
                    "evaluation_status": "diagnostic_gold_v0_small_unvalidated",
                    "case_count": 10,
                    "mismatches": [],
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "claim_diagnosis_report.md").write_text(
            "# Claim-Level Diagnostic Evaluation Report\n",
            encoding="utf-8",
        )
        assert env is not None
        assert env["PYTHONPATH"] == ".:src"
        return CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(freeze_module.subprocess, "run", _fake_run)

    result = freeze_day1_baseline(repo_root=repo_root, output_dir=tmp_path)

    tests_payload = json.loads((tmp_path / TEST_REPORT_NAME).read_text(encoding="utf-8"))
    claim_payload = json.loads((tmp_path / CLAIM_REPORT_JSON_NAME).read_text(encoding="utf-8"))

    assert tests_payload["command"] == "pytest -q"
    assert tests_payload["exit_code"] == 0
    assert tests_payload["summary"] == "339 passed in 38.54s"
    assert claim_payload["case_count"] == 10
    assert claim_payload["mismatches"] == []
    assert (tmp_path / CLAIM_REPORT_MD_NAME).exists()
    assert result.gold_case_count == 10


def test_freeze_day1_baseline_rejects_nonempty_claim_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    def _fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        if command == ["pytest", "-q"]:
            return CompletedProcess(command, 0, stdout="339 passed in 38.54s\n", stderr="")

        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "claim_diagnosis_report.json").write_text(
            json.dumps(
                {
                    "evaluation_status": "diagnostic_gold_v0_small_unvalidated",
                    "case_count": 10,
                    "mismatches": [{"case_id": "unsupported_missing_1"}],
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "claim_diagnosis_report.md").write_text("report\n", encoding="utf-8")
        return CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(freeze_module.subprocess, "run", _fake_run)

    with pytest.raises(ValueError, match="mismatches"):
        freeze_day1_baseline(repo_root=repo_root, output_dir=tmp_path)
