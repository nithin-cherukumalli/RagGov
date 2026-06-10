"""Audit-only preflight for safe GovRAG coding-agent edits."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import (
    ROOT,
    branch,
    git_status,
    last_commit,
    protected_changes,
    recent_report_status,
    run_command,
    status_from_messages,
    summarize_command_result,
    threshold_or_gate_changes,
    write_json,
    write_markdown_report,
)


def build_report(run_common: bool = False, strict: bool = False, cwd: Path = ROOT) -> dict:
    status = git_status(cwd)
    dirty_files = status["dirty_files"]
    protected = protected_changes(dirty_files, cwd)
    gates = threshold_or_gate_changes(dirty_files, cwd)
    report_status = recent_report_status(cwd)
    warnings: list[str] = []
    errors: list[str] = []
    benchmark_summary: dict = {"run_common": run_common}

    if dirty_files:
        warnings.append("Workspace has dirty files before edits.")
    if status["deleted_tracked_files"]:
        errors.append("Tracked files are deleted.")
    if protected:
        warnings.append("Protected benchmark, fixture, golden, or report paths are changed.")
    if gates:
        warnings.append("Threshold, launch-readiness, or production-gating related files are changed.")
    missing_reports = [key for key, exists in report_status.items() if not exists]
    if missing_reports:
        warnings.append("Recent common-failure or launch-readiness reports are missing.")

    commands_run: list[dict] = []
    if run_common:
        result = run_command(["python", "scripts/evaluate_common_failures.py"], cwd=cwd, timeout=300)
        commands_run.append(summarize_command_result(result))
        benchmark_summary["evaluate_common_failures_returncode"] = result["returncode"]
        if result["returncode"] != 0:
            warnings.append("Common benchmark command returned non-zero.")

    report = {
        "status": status_from_messages(errors, warnings, strict=strict),
        "branch": branch(cwd),
        "last_commit": last_commit(cwd),
        "dirty_files": dirty_files,
        "deleted_tracked_files": status["deleted_tracked_files"],
        "untracked_files": status["untracked_files"],
        "protected_changes": protected,
        "threshold_or_gate_changes": gates,
        "benchmark_summary": benchmark_summary,
        "recent_reports": report_status,
        "commands_run": commands_run,
        "warnings": warnings,
        "errors": errors,
        "recommended_action": (
            "Stop and resolve errors before edits."
            if errors
            else "Review warnings, then proceed only with a narrow harness-safe patch."
            if warnings
            else "Proceed with the requested narrow patch."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-common", action="store_true", help="Run the common benchmark command.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args(argv)
    report = build_report(run_common=args.run_common, strict=args.strict, cwd=ROOT)
    write_json(ROOT / "reports" / "harness_preflight_report.json", report)
    write_markdown_report(ROOT / "reports" / "harness_preflight_report.md", "Harness Preflight Report", report)
    print(f"harness preflight: {report['status']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
