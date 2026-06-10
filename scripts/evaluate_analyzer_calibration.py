#!/usr/bin/env python3
"""Generate audit-only analyzer-wise calibration/readiness reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from raggov.evaluation.analyzer_calibration import (  # noqa: E402
    AnalyzerCalibrationCase,
    compute_analyzer_calibration_metrics,
    render_analyzer_calibration_markdown,
)
from stresslab.cases.load import load_common_rag_failures, load_subtle_rag_failures  # noqa: E402
from stresslab.runners.rag_failure_runner import RAGFailureRunner  # noqa: E402


def collect_cases(*, suites: list[str], modes: list[str]) -> list[AnalyzerCalibrationCase]:
    records: list[AnalyzerCalibrationCase] = []
    for suite in suites:
        golden_cases = load_subtle_rag_failures() if suite == "subtle" else load_common_rag_failures()
        for mode in modes:
            runner = RAGFailureRunner(mode=mode, suite=suite)
            for case in golden_cases:
                run = runner._build_run(case)
                diagnosis = runner.engine.diagnose(run)
                trace = diagnosis.diagnosis_decision_trace or {}
                records.append(
                    AnalyzerCalibrationCase(
                        case_id=case.case_id,
                        category=case.category,
                        mode=mode,
                        expected_primary=case.expected_primary_failure,
                        expected_stage=case.expected_root_cause_stage,
                        actual_primary=diagnosis.primary_failure.value,
                        actual_stage=diagnosis.root_cause_stage.value,
                        selected_analyzer=trace.get("selected_analyzer"),
                        analyzer_results=diagnosis.analyzer_results,
                    )
                )
    return records


def build_audit_payload(*, suites: list[str], modes: list[str]) -> dict[str, Any]:
    return {
        "audit_title": "Analyzer-wise Calibration Evaluator Audit",
        "audit_only": True,
        "runtime_behavior_changed": False,
        "benchmark_labels_changed": False,
        "thresholds_changed": False,
        "decision_policy_changed": False,
        "production_gating_enabled": False,
        "calibration_claim_made": False,
        "suites": suites,
        "modes": modes,
        "metrics": [
            "candidate_count",
            "selected_count",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "precision",
            "recall",
            "stage_accuracy",
            "failure_type_accuracy",
            "false_clean_contribution",
            "evidence_strength_distribution",
            "method_status_distribution",
            "calibration_status_distribution",
        ],
        "method_notes": [
            "Precision and recall are descriptive benchmark audit metrics, not calibrated confidence.",
            "False negatives count cases where an analyzer emitted the exact expected failure/stage candidate but another analyzer was selected incorrectly.",
            "Unclaimed expected failures count failed cases where no analyzer emitted the exact expected failure/stage candidate.",
        ],
    }


def render_audit_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Analyzer-wise Calibration Evaluator Audit",
            "",
            "This evaluator is audit-only and readiness-supporting.",
            "",
            "## Scope",
            "",
            f"- Suites: `{payload['suites']}`",
            f"- Modes: `{payload['modes']}`",
            "- Runtime behavior changed: `false`",
            "- Benchmark labels changed: `false`",
            "- Thresholds changed: `false`",
            "- Decision policy changed: `false`",
            "- Production gating enabled: `false`",
            "- Calibrated confidence claimed: `false`",
            "",
            "## Metrics",
            "",
            *[f"- `{metric}`" for metric in payload["metrics"]],
            "",
            "## Method Notes",
            "",
            *[f"- {note}" for note in payload["method_notes"]],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analyzer-wise calibration audit")
    parser.add_argument("--suite", choices=["common", "subtle", "all"], default="common")
    parser.add_argument(
        "--mode",
        choices=["native", "external-enhanced", "both"],
        default="both",
    )
    parser.add_argument("--audit-json", default="reports/analyzer_calibration_audit.json")
    parser.add_argument("--audit-md", default="reports/analyzer_calibration_audit.md")
    parser.add_argument("--result-json", default="reports/analyzer_calibration_result.json")
    parser.add_argument("--result-md", default="reports/analyzer_calibration_result.md")
    args = parser.parse_args()

    suites = ["common", "subtle"] if args.suite == "all" else [args.suite]
    modes = ["native", "external-enhanced"] if args.mode == "both" else [args.mode]

    audit_payload = build_audit_payload(suites=suites, modes=modes)
    records = collect_cases(suites=suites, modes=modes)
    result_payload = compute_analyzer_calibration_metrics(records)
    result_payload.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "suites_requested": suites,
            "modes_requested": modes,
            "source": "DiagnosisEngine benchmark analyzer_results",
        }
    )

    for path_str, content in (
        (args.audit_json, json.dumps(audit_payload, indent=2, sort_keys=True)),
        (args.audit_md, render_audit_markdown(audit_payload)),
        (args.result_json, json.dumps(result_payload, indent=2, sort_keys=True)),
        (args.result_md, render_analyzer_calibration_markdown(result_payload)),
    ):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    totals = result_payload["totals"]
    print(
        "Analyzer calibration audit completed: "
        f"{totals['passed_count']}/{totals['case_count']} cases passed "
        f"({totals['pass_rate']:.1%}); reports written under reports/."
    )


if __name__ == "__main__":
    main()
