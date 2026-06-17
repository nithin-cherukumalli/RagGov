"""Smoke tests for the engineer-facing `raggov diagnose` CLI.

Day 1 of v0.1-alpha-public sprint. These tests verify the contract that
the CLI presents to a real engineer:

* exits 0 on valid input,
* prints a one-page report,
* makes the calibration posture visible via a mandatory footer.

These are deliberately minimal so they do not pin the report layout
beyond the engineer-facing contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI via `python -m raggov.cli` so tests do not require an
    editable install. Honest tradeoff: this exercises the same code path
    as the installed `raggov` entry point but skips the entry-point shim.
    """
    return subprocess.run(
        [sys.executable, "-m", "raggov.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_diagnose_text_format_clean_pass_exits_zero_and_shows_clean() -> None:
    """CLEAN fixture must produce a CLEAN diagnosis and the mandated footer."""
    result = _run_cli(
        "diagnose",
        str(FIXTURES / "clean_pass.json"),
        "--mode",
        "native",
        "--format",
        "text",
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Primary Failure: CLEAN" in result.stdout
    # Mandated footer disclosure: an engineer skimming the report must see
    # that this is not a calibrated production verdict.
    assert "mode=native" in result.stdout
    assert "calibration_status=" in result.stdout
    assert "production_gating_eligible=False" in result.stdout


def test_diagnose_text_format_unsupported_claims_escalates_to_human_review() -> None:
    """UNSUPPORTED_CLAIM fixture must escalate to human review with footer."""
    result = _run_cli(
        "diagnose",
        str(FIXTURES / "unsupported_claims.json"),
        "--mode",
        "native",
        "--format",
        "text",
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Primary Failure: UNSUPPORTED_CLAIM" in result.stdout
    assert "Human Review Required: True" in result.stdout
    assert "production_gating_eligible=False" in result.stdout


def test_diagnose_json_format_emits_valid_json_to_stdout() -> None:
    """`--format json` must emit a parseable JSON document with the
    diagnosis dictionary, with no side-effect files.
    """
    result = _run_cli(
        "diagnose",
        str(FIXTURES / "clean_pass.json"),
        "--mode",
        "native",
        "--format",
        "json",
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["primary_failure"] == "CLEAN"
    assert "run_id" in payload


def test_diagnose_json_format_includes_why_block_for_citation_mismatch() -> None:
    result = _run_cli(
        "diagnose",
        str(FIXTURES / "citation_mismatch.json"),
        "--mode",
        "native",
        "--format",
        "json",
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    why_block = payload["why_block"]
    assert set(why_block) == {
        "verdict_summary",
        "voted_by",
        "also_considered",
        "inspect_next",
    }
    assert isinstance(why_block["voted_by"], list)
    assert isinstance(why_block["also_considered"], list)
    assert any(
        item["analyzer"] and item["failure_type"] and item["status"]
        for item in why_block["also_considered"]
    )


def test_diagnose_rejects_invalid_format_with_nonzero_exit() -> None:
    result = _run_cli(
        "diagnose",
        str(FIXTURES / "clean_pass.json"),
        "--mode",
        "native",
        "--format",
        "yaml",
    )

    assert result.returncode != 0
