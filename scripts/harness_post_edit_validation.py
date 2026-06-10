"""Post-edit validation for GovRAG coding-agent changes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from harness_common import (
    ROOT,
    baseline_config,
    classify_risk,
    git_status,
    load_common_report,
    load_launch_report,
    common_benchmark_mode_warnings,
    find_key,
    parse_common_benchmark_modes,
    protected_changes,
    run_command,
    status_from_messages,
    summarize_command_result,
    threshold_or_gate_changes,
    write_json,
    write_markdown_report,
)


def _find_gating(common_report: dict, launch_report: dict) -> bool | None:
    value = find_key(launch_report, "production_gating_eligible")
    if value is None:
        value = find_key(common_report, "production_gating_eligible")
    return value if isinstance(value, bool) else None


def _regression(metric: str, baseline: Any, current: Any, severity: str, message: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "severity": severity,
        "message": message,
    }


def _observed_max(mode_results: dict[str, dict[str, int | None] | None], key: str) -> int | None:
    values = [
        mode.get(key)
        for mode in mode_results.values()
        if isinstance(mode, dict) and isinstance(mode.get(key), int)
    ]
    return max(values) if values else None


def _baseline_regressions(
    common_report: dict, launch_report: dict, baseline: dict
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    regressions: list[dict[str, Any]] = []
    warnings = []
    observed: dict[str, Any] = {}
    mode_results = parse_common_benchmark_modes(common_report)
    observed["mode_results"] = mode_results
    warnings.extend(common_benchmark_mode_warnings(mode_results))

    mode_specs = {
        "native": {
            "passed_metric": "common_native_passed",
            "total_metric": "common_native_total",
            "label": "Native",
        },
        "external_enhanced": {
            "passed_metric": "common_external_passed",
            "total_metric": "common_external_total",
            "label": "External-enhanced",
        },
    }
    for mode_name, spec in mode_specs.items():
        mode = mode_results.get(mode_name)
        if not isinstance(mode, dict):
            continue
        passed = mode.get("passed")
        baseline_passed = baseline.get(spec["passed_metric"])
        if passed is None:
            warnings.append(f"{spec['label']} common benchmark pass count is missing.")
        elif baseline_passed is not None and passed < baseline_passed:
            regressions.append(
                _regression(
                    spec["passed_metric"],
                    baseline_passed,
                    passed,
                    "high",
                    f"{spec['label']} common benchmark pass count regressed below protected baseline.",
                )
            )

        total = mode.get("total")
        baseline_total = baseline.get(spec["total_metric"])
        if total is None:
            warnings.append(f"{spec['label']} common benchmark total count is missing.")
        elif baseline_total is not None and total != baseline_total:
            regressions.append(
                _regression(
                    spec["total_metric"],
                    baseline_total,
                    total,
                    "high",
                    f"{spec['label']} common benchmark total changed from protected baseline.",
                )
            )

        for key in ("false_clean_count", "false_security_count", "false_incomplete_count"):
            current = mode.get(key)
            expected = baseline.get(key)
            metric = f"{mode_name}_{key}"
            if current is None:
                warnings.append(f"{spec['label']} {key} is missing.")
            elif expected is not None and current > expected:
                regressions.append(
                    _regression(
                        metric,
                        expected,
                        current,
                        "high",
                        f"{spec['label']} {key} increased above protected baseline.",
                    )
                )

    for key in ("false_clean_count", "false_security_count", "false_incomplete_count"):
        observed[key] = _observed_max(mode_results, key)

    observed["production_gating_eligible"] = _find_gating(common_report, launch_report)
    if observed["production_gating_eligible"] is True:
        regressions.append(
            _regression(
                "production_gating_eligible",
                False,
                True,
                "high",
                "production_gating_eligible is true",
            )
        )
    return regressions, observed, warnings


def build_report(
    run_common: bool = False,
    run_launch: bool = False,
    strict: bool = False,
    cwd: Path = ROOT,
) -> dict:
    status = git_status(cwd)
    changed_files = status["dirty_files"]
    risk = classify_risk(changed_files, cwd)
    protected = protected_changes(changed_files, cwd)
    gates = threshold_or_gate_changes(changed_files, cwd)
    warnings: list[str] = []
    errors: list[str] = []
    commands_run: list[dict] = []

    if run_common:
        result = run_command(["python", "scripts/evaluate_common_failures.py"], cwd=cwd, timeout=300)
        commands_run.append(summarize_command_result(result))
        if result["returncode"] != 0:
            warnings.append("Common benchmark command returned non-zero.")
    if run_launch:
        result = run_command(["python", "scripts/launch_readiness.py"], cwd=cwd, timeout=300)
        commands_run.append(summarize_command_result(result))
        if result["returncode"] != 0:
            warnings.append("Launch-readiness command returned non-zero.")

    common_report = load_common_report(cwd)
    launch_report = load_launch_report(cwd)
    baseline = baseline_config(cwd)
    regressions, observed, baseline_warnings = _baseline_regressions(common_report, launch_report, baseline)
    warnings.extend(baseline_warnings)

    if protected:
        warnings.append("Protected benchmark, fixture, golden, or report paths are changed.")
    if gates:
        warnings.append("Threshold, launch-readiness, or production-gating related files are changed.")
    if risk["critical"]:
        warnings.append("Critical-risk files changed.")
    if regressions:
        errors.extend(regression["message"] for regression in regressions)

    report = {
        "status": status_from_messages(errors, warnings, strict=strict),
        "changed_files": changed_files,
        "risk_classification": risk,
        "commands_run": commands_run,
        "benchmark_before_or_baseline": baseline,
        "benchmark_after": common_report,
        "mode_results": observed.get("mode_results", {"native": None, "external_enhanced": None}),
        "false_clean_count": observed.get("false_clean_count"),
        "false_security_count": observed.get("false_security_count"),
        "false_incomplete_count": observed.get("false_incomplete_count"),
        "production_gating_eligible": observed.get("production_gating_eligible", False),
        "protected_changes": protected,
        "threshold_or_gate_changes": gates,
        "protected_baseline_regressions": regressions,
        "warnings": warnings,
        "errors": errors,
        "recommended_action": (
            "Stop and investigate baseline regressions or protected edits."
            if errors or protected or gates
            else "Proceed with targeted tests and final reporting."
            if warnings
            else "Post-edit validation passed."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-common", action="store_true", help="Run the common benchmark command.")
    parser.add_argument("--run-launch", action="store_true", help="Run launch-readiness command.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args(argv)
    report = build_report(
        run_common=args.run_common,
        run_launch=args.run_launch,
        strict=args.strict,
        cwd=ROOT,
    )
    write_json(ROOT / "reports" / "harness_post_edit_validation.json", report)
    write_markdown_report(
        ROOT / "reports" / "harness_post_edit_validation.md",
        "Harness Post-Edit Validation",
        report,
    )
    print(f"harness post-edit validation: {report['status']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
