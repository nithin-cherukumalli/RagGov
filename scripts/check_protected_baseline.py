#!/usr/bin/env python3
"""Run and verify the protected GovRAG common benchmark baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRIAGE_PATH = ROOT / "reports" / "common_failure_triage.json"

EXPECTED_FAILURES = {
    "retrieval_top_k_too_small_08",
    "retrieval_irrelevant_plausible_09",
    "security_retrieval_anomaly_only_36",
    "quality_incomplete_38",
    "quality_ignores_context_41",
}

EXPECTED_CATEGORY_PASSES = {
    "citation": (5, 5),
    "grounding": (7, 7),
    "sufficiency": (5, 5),
    "version_validity": (5, 5),
}


def _run(args: list[str]) -> int:
    result = subprocess.run(args, cwd=ROOT, text=True, check=False)
    return result.returncode


def _load_triage() -> dict[str, Any]:
    return json.loads(TRIAGE_PATH.read_text(encoding="utf-8"))


def _mode_errors(mode_name: str, mode: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    passed = mode.get("passed_cases")
    total = mode.get("total_cases")
    if (passed, total) != (41, 46):
        errors.append(f"{mode_name}: expected 41/46, got {passed}/{total}")

    for key in ("false_clean_count", "false_security_count", "false_incomplete_count"):
        if mode.get(key) != 0:
            errors.append(f"{mode_name}: expected {key}=0, got {mode.get(key)}")

    failures = {failure.get("case_id") for failure in mode.get("failures", [])}
    if failures != EXPECTED_FAILURES:
        errors.append(
            f"{mode_name}: expected failures {sorted(EXPECTED_FAILURES)}, got {sorted(failures)}"
        )

    category_stats = mode.get("category_stats", {})
    for category, expected in EXPECTED_CATEGORY_PASSES.items():
        stats = category_stats.get(category, {})
        actual = (stats.get("passed"), stats.get("total"))
        if actual != expected:
            errors.append(f"{mode_name}: expected {category} {expected[0]}/{expected[1]}, got {actual[0]}/{actual[1]}")

    return errors


def main() -> int:
    preflight = _run(["python", "scripts/harness_preflight.py"])
    benchmark = _run(["python", "scripts/evaluate_common_failures.py", "--suite", "common"])
    if benchmark != 0:
        print("protected baseline check: benchmark command failed", file=sys.stderr)
        return benchmark

    triage = _load_triage()
    modes = triage.get("modes", {})
    errors: list[str] = []
    for mode_name in ("native", "external-enhanced"):
        mode = modes.get(mode_name)
        if not isinstance(mode, dict):
            errors.append(f"{mode_name}: mode result missing")
            continue
        errors.extend(_mode_errors(mode_name, mode))

    if errors:
        print("protected baseline check: fail")
        for error in errors:
            print(f"- {error}")
        return 1

    if preflight != 0:
        print("protected baseline check: pass with preflight warnings")
        return 0

    print("protected baseline check: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
