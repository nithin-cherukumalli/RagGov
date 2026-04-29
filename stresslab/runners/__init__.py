"""Thin runner helpers for ingest and index artifact generation."""

from .build_index import BuildIndexRunResult, run_build_index
from .freeze_day1_baseline import BaselineFreezeResult, freeze_day1_baseline
from .ingest import IngestRunResult, run_ingest
from .run_case import RunCaseResult, run_case
from .run_claim_diagnosis_harness import (
    run_claim_diagnosis_suite,
    write_claim_diagnosis_markdown_report,
    write_claim_diagnosis_report,
)
from .run_diagnosis_suite import (
    DiagnosisGoldenCaseSummary,
    DiagnosisGoldenSuiteResult,
    render_diagnosis_suite_markdown,
    run_diagnosis_suite,
    write_diagnosis_suite_markdown_report,
    write_diagnosis_suite_report,
)
from .run_suite import (
    RunSuiteCaseSummary,
    RunSuiteResult,
    render_suite_markdown,
    run_suite,
    write_suite_markdown_report,
    write_suite_report,
)

__all__ = [
    "BuildIndexRunResult",
    "BaselineFreezeResult",
    "DiagnosisGoldenCaseSummary",
    "DiagnosisGoldenSuiteResult",
    "IngestRunResult",
    "RunCaseResult",
    "RunSuiteCaseSummary",
    "RunSuiteResult",
    "freeze_day1_baseline",
    "render_diagnosis_suite_markdown",
    "render_suite_markdown",
    "run_build_index",
    "run_case",
    "run_claim_diagnosis_suite",
    "run_diagnosis_suite",
    "run_ingest",
    "run_suite",
    "write_claim_diagnosis_markdown_report",
    "write_claim_diagnosis_report",
    "write_diagnosis_suite_markdown_report",
    "write_diagnosis_suite_report",
    "write_suite_markdown_report",
    "write_suite_report",
]
