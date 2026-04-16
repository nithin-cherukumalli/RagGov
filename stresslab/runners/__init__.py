"""Thin runner helpers for ingest and index artifact generation."""

from .build_index import BuildIndexRunResult, run_build_index
from .ingest import IngestRunResult, run_ingest
from .run_case import RunCaseResult, run_case
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
    "DiagnosisGoldenCaseSummary",
    "DiagnosisGoldenSuiteResult",
    "IngestRunResult",
    "RunCaseResult",
    "RunSuiteCaseSummary",
    "RunSuiteResult",
    "render_diagnosis_suite_markdown",
    "render_suite_markdown",
    "run_build_index",
    "run_case",
    "run_diagnosis_suite",
    "run_ingest",
    "run_suite",
    "write_diagnosis_suite_markdown_report",
    "write_diagnosis_suite_report",
    "write_suite_markdown_report",
    "write_suite_report",
]
