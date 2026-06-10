#!/usr/bin/env python3
"""Generate audit-only subtle suite failure triage reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from raggov.evaluation.analyzer_calibration import AnalyzerCalibrationCase
from raggov.evaluation.subtle_failure_triage import compute_triage_audit, render_triage_markdown
from stresslab.cases.load import load_subtle_rag_failures, load_common_rag_failures
from stresslab.runners.rag_failure_runner import RAGFailureRunner

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Run subtle failure triage audit")
    parser.add_argument("--suite", choices=["common", "subtle", "all"], default="subtle")
    parser.add_argument("--mode", choices=["native", "external-enhanced", "both"], default="both")
    parser.add_argument("--audit-json", default="reports/subtle_failure_triage_audit.json")
    parser.add_argument("--audit-md", default="reports/subtle_failure_triage_audit.md")
    parser.add_argument("--result-json", default="reports/subtle_failure_triage_result.json")
    parser.add_argument("--result-md", default="reports/subtle_failure_triage_result.md")
    args = parser.parse_args()

    suites = ["common", "subtle"] if args.suite == "all" else [args.suite]
    modes = ["native", "external-enhanced"] if args.mode == "both" else [args.mode]

    records = collect_cases(suites=suites, modes=modes)
    payload = compute_triage_audit(records)
    
    for path_str, content in (
        (args.audit_json, json.dumps(payload, indent=2, sort_keys=True)),
        (args.audit_md, render_triage_markdown(payload)),
        (args.result_json, json.dumps(payload, indent=2, sort_keys=True)),
        (args.result_md, render_triage_markdown(payload)),
    ):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    print(f"Subtle failure triage audit completed: {payload['total_failures']} failures logged.")

if __name__ == "__main__":
    main()
