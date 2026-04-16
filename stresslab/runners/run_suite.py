"""Batch stresslab runner."""

from __future__ import annotations

from dataclasses import dataclass

from .run_case import run_case


@dataclass(frozen=True)
class RunSuiteCaseSummary:
    case_id: str
    run_id: str
    primary_failure: str
    should_have_answered: bool


@dataclass(frozen=True)
class RunSuiteResult:
    results: list[RunSuiteCaseSummary]
    total_count: int


def run_suite(case_ids: list[str], profile: str, dry_run: bool = False) -> RunSuiteResult:
    results = []
    for case_id in case_ids:
        case_result = run_case(case_id=case_id, profile=profile, dry_run=dry_run)
        results.append(
            RunSuiteCaseSummary(
                case_id=case_result.case_id,
                run_id=case_result.run.run_id,
                primary_failure=case_result.diagnosis.primary_failure.value,
                should_have_answered=case_result.diagnosis.should_have_answered,
            )
        )
    return RunSuiteResult(results=results, total_count=len(results))
