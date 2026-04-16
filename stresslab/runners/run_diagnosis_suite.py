"""Diagnosis-native golden suite runner."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from raggov import diagnose_file

from stresslab.cases import load_diagnosis_golden_case
from stresslab.diagnosis_evaluation import evaluate_diagnosis_case
from stresslab.reports import write_json_artifact


_ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiagnosisGoldenCaseSummary:
    case_id: str
    run_fixture: str
    primary_failure: str
    root_cause_stage: str
    should_have_answered: bool
    expectation_matched: bool = False
    expectation_notes: list[str] | None = None


@dataclass(frozen=True)
class DiagnosisGoldenSuiteResult:
    results: list[DiagnosisGoldenCaseSummary]
    total_count: int
    matched_count: int = 0
    match_rate: float = 0.0
    observed_primary_failures: dict[str, int] | None = None
    observed_root_cause_stages: dict[str, int] | None = None
    mismatched_case_ids: list[str] | None = None
    recurring_mismatch_notes: list[str] | None = None


def run_diagnosis_suite(case_ids: list[str]) -> DiagnosisGoldenSuiteResult:
    """Run diagnosis-native golden cases against fixture-backed RagGov runs."""
    results: list[DiagnosisGoldenCaseSummary] = []
    matched_count = 0
    primary_failures: Counter[str] = Counter()
    root_cause_stages: Counter[str] = Counter()
    mismatch_notes: Counter[str] = Counter()
    mismatched_case_ids: list[str] = []

    for case_id in case_ids:
        case = load_diagnosis_golden_case(case_id)
        diagnosis = diagnose_file(_resolve_run_fixture(case.run_fixture), config=case.engine_config)
        evaluation = evaluate_diagnosis_case(case, diagnosis)

        primary_failures[diagnosis.primary_failure.value] += 1
        root_cause_stages[diagnosis.root_cause_stage.value] += 1
        if evaluation.matched_overall:
            matched_count += 1
        else:
            mismatched_case_ids.append(case.case_id)
            mismatch_notes.update(evaluation.notes)

        results.append(
            DiagnosisGoldenCaseSummary(
                case_id=case.case_id,
                run_fixture=case.run_fixture,
                primary_failure=diagnosis.primary_failure.value,
                root_cause_stage=diagnosis.root_cause_stage.value,
                should_have_answered=diagnosis.should_have_answered,
                expectation_matched=evaluation.matched_overall,
                expectation_notes=evaluation.notes,
            )
        )

    total_count = len(results)
    return DiagnosisGoldenSuiteResult(
        results=results,
        total_count=total_count,
        matched_count=matched_count,
        match_rate=(matched_count / total_count) if total_count else 0.0,
        observed_primary_failures=dict(primary_failures),
        observed_root_cause_stages=dict(root_cause_stages),
        mismatched_case_ids=mismatched_case_ids,
        recurring_mismatch_notes=[note for note, _ in mismatch_notes.most_common(10)],
    )


def write_diagnosis_suite_report(
    result: DiagnosisGoldenSuiteResult,
    output_path: str | Path,
) -> Path:
    """Write a diagnosis-golden suite JSON report artifact."""
    payload = {
        "total_count": result.total_count,
        "matched_count": result.matched_count,
        "match_rate": result.match_rate,
        "observed_primary_failures": result.observed_primary_failures or {},
        "observed_root_cause_stages": result.observed_root_cause_stages or {},
        "mismatched_case_ids": result.mismatched_case_ids or [],
        "recurring_mismatch_notes": result.recurring_mismatch_notes or [],
        "results": [asdict(summary) for summary in result.results],
    }
    return write_json_artifact(output_path, payload)


def render_diagnosis_suite_markdown(result: DiagnosisGoldenSuiteResult) -> str:
    """Render a markdown summary for diagnosis-golden suite results."""
    lines = [
        "# Diagnosis Golden Suite Report",
        "",
        f"- Total cases: {result.total_count}",
        f"- Matched cases: {result.matched_count}",
        f"- Match rate: {result.match_rate:.0%}",
        "",
        "## Observed Primary Failures",
    ]

    observed_primary_failures = result.observed_primary_failures or {}
    if observed_primary_failures:
        for failure, count in sorted(observed_primary_failures.items()):
            lines.append(f"- `{failure}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Observed Root Cause Stages"])
    observed_stages = result.observed_root_cause_stages or {}
    if observed_stages:
        for stage, count in sorted(observed_stages.items()):
            lines.append(f"- `{stage}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Mismatches"])
    mismatched_case_ids = result.mismatched_case_ids or []
    if mismatched_case_ids:
        for case_id in mismatched_case_ids:
            lines.append(f"- `{case_id}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Recurring Mismatch Notes"])
    recurring_notes = result.recurring_mismatch_notes or []
    if recurring_notes:
        for note in recurring_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_diagnosis_suite_markdown_report(
    result: DiagnosisGoldenSuiteResult,
    output_path: str | Path,
) -> Path:
    """Write a markdown summary artifact for a diagnosis-golden suite run."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_diagnosis_suite_markdown(result), encoding="utf-8")
    return target


def _resolve_run_fixture(run_fixture: str) -> Path:
    path = Path(run_fixture)
    if path.is_absolute():
        return path
    return (_ROOT_DIR / path).resolve()
