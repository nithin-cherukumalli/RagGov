"""Check the minimum v0.1-alpha release-freeze gates.

This script intentionally does not require full pytest, production calibration,
or external provider availability. It runs each heavy gate in a subprocess and
collects only JSON summaries, so it does not rewrite benchmark or launch
readiness reports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PASSED = 41
PROTECTED_TOTAL = 46
SUMMARY_PREFIX = "ALPHA_GATE_JSON="


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: str


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    results: list[GateResult] = []

    native_report = _run_common("native")
    external_report = _run_common("external-enhanced")
    results.extend(_benchmark_gates("common_native", native_report))
    results.extend(_benchmark_gates("common_external_enhanced", external_report))

    launch_report = _run_launch_readiness()
    launch_metrics = launch_report["metrics"]
    results.append(
        GateResult(
            "launch readiness alpha status",
            "v0.1-alpha-clean Ready" in launch_report["status"]
            and launch_report["passed"],
            f"status={launch_report['status']!r}; passed={launch_report['passed']}",
        )
    )
    results.append(
        GateResult(
            "production gating remains disabled",
            launch_metrics.get("production_gating_eligible") is False,
            f"production_gating_eligible={launch_metrics.get('production_gating_eligible')}",
        )
    )
    results.append(
        GateResult(
            "production remains not ready",
            launch_metrics.get("production_readiness_status") == "Not Ready",
            f"production_readiness_status={launch_metrics.get('production_readiness_status')!r}",
        )
    )

    harness = _run_harness_tests()
    results.append(
        GateResult(
            "harness tests pass",
            harness.returncode == 0,
            f"exit_code={harness.returncode}",
        )
    )

    _print_results(results)
    failures = [result for result in results if not result.passed]
    if failures:
        print("\nALPHA RELEASE GATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure.name}: {failure.details}", file=sys.stderr)
        return 1

    print("\nALPHA RELEASE GATE PASSED")
    return 0


def _run_common(mode: str) -> dict[str, Any]:
    code = f"""
import json
from stresslab.runners.rag_failure_runner import RAGFailureRunner

report = RAGFailureRunner(mode={mode!r}, suite="common").run_benchmark()
payload = {{
    "passed_cases": report.passed_cases,
    "total_cases": report.total_cases,
    "false_clean_count": sum(
        1 for result in report.results
        if result.expected_primary != "CLEAN" and result.actual_primary == "CLEAN"
    ),
    "false_security_count": sum(
        1 for result in report.results
        if result.actual_stage == "SECURITY" and result.expected_stage != "SECURITY"
    ),
    "false_incomplete_count": sum(
        1 for result in report.results
        if result.expected_primary != "INCOMPLETE_DIAGNOSIS"
        and result.actual_primary == "INCOMPLETE_DIAGNOSIS"
    ),
}}
print({SUMMARY_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    return _run_json_subprocess([sys.executable, "-c", code])


def _run_launch_readiness() -> dict[str, Any]:
    code = f"""
import json
import os
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
from stresslab.runners.launch_readiness import run_launch_readiness

report = run_launch_readiness()
payload = {{
    "status": report.status,
    "passed": report.passed,
    "metrics": {{
        "production_gating_eligible": report.metrics.get("production_gating_eligible"),
        "production_readiness_status": report.metrics.get("production_readiness_status"),
    }},
}}
print({SUMMARY_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    return _run_json_subprocess([sys.executable, "-c", code])


def _run_json_subprocess(command: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(SUMMARY_PREFIX):
            return json.loads(line.removeprefix(SUMMARY_PREFIX))
    stderr_tail = "\n".join(completed.stderr.splitlines()[-10:])
    stdout_tail = "\n".join(completed.stdout.splitlines()[-10:])
    raise RuntimeError(
        "gate subprocess did not emit a JSON summary; "
        f"exit_code={completed.returncode}; stdout_tail={stdout_tail!r}; "
        f"stderr_tail={stderr_tail!r}"
    )


def _benchmark_gates(prefix: str, report: dict[str, Any]) -> list[GateResult]:
    false_clean = report["false_clean_count"]
    false_security = report["false_security_count"]
    false_incomplete = report["false_incomplete_count"]
    return [
        GateResult(
            f"{prefix} protected pass count",
            report["passed_cases"] >= PROTECTED_PASSED
            and report["total_cases"] == PROTECTED_TOTAL,
            f"{report['passed_cases']}/{report['total_cases']}",
        ),
        GateResult(f"{prefix} false_clean_count == 0", false_clean == 0, str(false_clean)),
        GateResult(
            f"{prefix} false_security_count == 0",
            false_security == 0,
            str(false_security),
        ),
        GateResult(
            f"{prefix} false_incomplete_count == 0",
            false_incomplete == 0,
            str(false_incomplete),
        ),
    ]


def _run_harness_tests() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/harness"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def _print_results(results: list[GateResult]) -> None:
    print("v0.1-alpha release gate")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"- {status}: {result.name} ({result.details})")


if __name__ == "__main__":
    raise SystemExit(main())
