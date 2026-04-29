"""Freeze a verified Day 1 baseline checkpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stresslab.cases import GOLDEN_DIR, load_claim_diagnosis_gold_set
from stresslab.reports import write_json_artifact


TEST_REPORT_NAME = "baseline_after_day1_tests.json"
CLAIM_REPORT_JSON_NAME = "baseline_after_day1_claim_report.json"
CLAIM_REPORT_MD_NAME = "baseline_after_day1_claim_report.md"
CLAIM_GOLD_SET_PATH = Path("stresslab/cases/golden/claim_diagnosis_gold_v0.json")
CLAIM_EVALUATION_STATUS = "diagnostic_gold_v0_small_unvalidated"


@dataclass(frozen=True)
class BaselineFreezeResult:
    """Summary of generated baseline artifacts."""

    tests_report_path: Path
    claim_report_json_path: Path
    claim_report_md_path: Path
    pytest_summary: str
    gold_case_count: int


def freeze_day1_baseline(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> BaselineFreezeResult:
    """Run the baseline-freeze workflow and write checkpoint artifacts."""
    root = Path(repo_root).resolve()
    target_dir = root if output_dir is None else Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    pytest_result = _run_command(["pytest", "-q"], cwd=root)
    pytest_summary = _extract_pytest_summary(pytest_result.stdout)
    tests_report_path = write_json_artifact(
        target_dir / TEST_REPORT_NAME,
        {
            "command": "pytest -q",
            "exit_code": pytest_result.returncode,
            "stdout": pytest_result.stdout,
            "stderr": pytest_result.stderr,
            "summary": pytest_summary,
        },
    )
    if pytest_result.returncode != 0:
        raise RuntimeError("pytest -q failed")

    with tempfile.TemporaryDirectory(prefix="baseline_after_day1_", dir=target_dir) as temp_dir:
        temp_output_dir = Path(temp_dir)
        claim_result = _run_command(
            ["raggov", "stresslab-claim-diagnosis", "--format", "both", "--output-dir", str(temp_output_dir)],
            cwd=root,
            env={**os.environ, "PYTHONPATH": ".:src"},
        )
        if claim_result.returncode != 0:
            raise RuntimeError("stresslab claim-diagnosis harness failed")

        generated_json = temp_output_dir / "claim_diagnosis_report.json"
        generated_md = temp_output_dir / "claim_diagnosis_report.md"
        claim_payload = _validate_claim_report(generated_json)

        claim_report_json_path = target_dir / CLAIM_REPORT_JSON_NAME
        claim_report_md_path = target_dir / CLAIM_REPORT_MD_NAME
        shutil.copyfile(generated_json, claim_report_json_path)
        shutil.copyfile(generated_md, claim_report_md_path)

    gold_set = load_claim_diagnosis_gold_set(CLAIM_GOLD_SET_PATH.name)
    gold_path = GOLDEN_DIR / CLAIM_GOLD_SET_PATH.name
    if len(gold_set.examples) != 10:
        raise ValueError(f"Expected 10 claim-diagnosis gold cases, found {len(gold_set.examples)}")
    if gold_path != (root / CLAIM_GOLD_SET_PATH).resolve():
        raise ValueError("Claim-diagnosis gold set path changed from expected location")

    return BaselineFreezeResult(
        tests_report_path=tests_report_path,
        claim_report_json_path=claim_report_json_path,
        claim_report_md_path=claim_report_md_path,
        pytest_summary=pytest_summary,
        gold_case_count=len(gold_set.examples),
    )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _extract_pytest_summary(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        summary = line.strip()
        if summary and " in " in summary and any(
            token in summary for token in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        ):
            return summary
    raise ValueError("Could not parse pytest summary from output")


def _validate_claim_report(report_path: Path) -> dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    if payload.get("evaluation_status") != CLAIM_EVALUATION_STATUS:
        raise ValueError(f"Unexpected claim evaluation status: {payload.get('evaluation_status')}")
    if payload.get("case_count") != 10:
        raise ValueError(f"Unexpected claim report case_count: {payload.get('case_count')}")

    mismatches = payload.get("mismatches")
    if mismatches != []:
        raise ValueError(f"Expected no claim report mismatches, found: {mismatches}")

    return payload
