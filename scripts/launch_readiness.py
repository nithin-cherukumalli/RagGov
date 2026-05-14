"""Run the GovRAG launch-readiness gate from a plain Python entry point."""

from __future__ import annotations

from pathlib import Path

from stresslab.runners import (
    LaunchReadinessReport,
    run_launch_readiness,
    write_launch_readiness_markdown_report,
    write_launch_readiness_report,
)


def main() -> int:
    try:
        report = run_launch_readiness()
    except Exception as exc:
        report = LaunchReadinessReport(
            status="Not Ready",
            passed=False,
            failure_reasons=[f"launch_readiness aborted: {exc}"],
            warnings=[],
            gates={"launch_readiness_runner_completed": False},
            metrics={
                "full_pytest_status": "not_run",
                "decision_policy_status": "not_run",
                "external_alignment_status": "not_run",
                "common_benchmark_pass_rate": 0.0,
                "subtle_benchmark_status": "aborted",
                "claim_harness_status": "not_run",
                "false_clean_count": 0,
                "false_security_count": 0,
                "external_ignored_count": 0,
                "missing_provider_reason_missing_count": 0,
                "calibrated_confidence_present_count": 0,
                "production_gating_eligible": False,
                "recommended_for_gating_true_count": 0,
            },
            checks={
                "launch_readiness_runner": {
                    "name": "launch_readiness_runner",
                    "passed": False,
                    "status": "aborted",
                    "error": str(exc),
                    "error_message": str(exc),
                    "launch_blocker": True,
                    "severity": "critical",
                    "remediation": "Inspect the launch-readiness runner exception and rerun scripts/launch_readiness.py after repair.",
                    "launch_blocker_classification": "code_test_health",
                }
            },
            launch_blockers=[
                {
                    "classification": "code_test_health",
                    "severity": "critical",
                    "message": f"launch_readiness aborted: {exc}",
                    "error": str(exc),
                    "remediation": "Inspect the launch-readiness runner exception and rerun scripts/launch_readiness.py after repair.",
                }
            ],
        )
    write_launch_readiness_report(report, Path("reports/launch_readiness_report.json"))
    write_launch_readiness_markdown_report(report, Path("reports/launch_readiness_report.md"))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
