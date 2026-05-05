"""Command-line interface entry points for RagGov."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from raggov.calibration import ARESCalibrator, ConfidenceInterval
from raggov.engine import DiagnosisEngine
from raggov.models.diagnosis import Diagnosis, FailureType, SecurityRisk
from raggov.models.run import RAGRun
from raggov.parser_validation.profile_questionnaire import profile_yaml_from_answers
from stresslab.cases import (
    list_cases as list_stresslab_cases,
    list_diagnosis_golden_cases,
)
from stresslab.runners import (
    run_claim_diagnosis_suite,
    run_diagnosis_suite,
    run_suite,
    write_claim_diagnosis_markdown_report,
    write_claim_diagnosis_report,
    write_diagnosis_suite_markdown_report,
    write_diagnosis_suite_report,
    write_suite_markdown_report,
    write_suite_report,
)


app = typer.Typer(help="Diagnose retrieval-augmented generation runs.")
console = Console()


@app.command()
def diagnose(
    run_file: Path,
    mode: str = typer.Option("external-enhanced", "--mode", help="Diagnosis mode: external-enhanced, native, or calibrated"),
) -> None:
    """Diagnose a RAGRun JSON file."""
    if mode == "calibrated":
        console.print("[red]Not Implemented[/red]: Calibrated mode is not yet available natively.")
        raise typer.Exit(code=1)

    try:
        run = _load_run(run_file)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        _print_validation_error(exc)
        raise typer.Exit(code=1) from exc

    engine_config = {"mode": mode}
    diagnosis = DiagnosisEngine(config=engine_config).diagnose(run)
    
    if diagnosis.degraded_external_mode:
        console.print(f"[yellow]Warning: Running in degraded external mode. Missing providers: {', '.join(diagnosis.missing_external_providers)}[/yellow]")

    console.print(_diagnosis_panel(diagnosis))

    output_path = Path.cwd() / f"{run.run_id}_diagnosis.json"
    output_path.write_text(diagnosis.model_dump_json(indent=2))
    console.print(f"[dim]Wrote raw diagnosis JSON to {output_path}[/dim]")


@app.command()
def validate(run_file: Path) -> None:
    """Validate a RAGRun JSON file."""
    try:
        _load_run(run_file)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        _print_validation_error(exc)
        raise typer.Exit(code=1) from exc

    console.print("✓ Valid RAGRun")


@app.command()
def calibrate(
    samples_file: Path,
    confidence: float = typer.Option(
        0.95, help="Confidence level (0.90, 0.95, or 0.99)"
    ),
) -> None:
    """Compute ARES PPI-corrected confidence intervals from calibration samples.

    This command implements Prediction-Powered Inference (PPI) to produce
    statistically valid confidence intervals on diagnostic metrics using
    human-labeled calibration samples.

    Requires at least 30 samples (recommended: 150-300).
    """
    try:
        calibrator = ARESCalibrator.load(samples_file, confidence_level=confidence)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        if isinstance(exc, OSError):
            console.print(f"[red]Error loading samples[/red]: {exc}")
        else:
            console.print(f"[red]Invalid calibration samples[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        intervals = calibrator.calibrate()
    except ValueError as exc:
        console.print(f"[red]Calibration error[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(_calibration_panel(intervals, len(calibrator._samples)))

    # Save report as JSON
    output_path = Path.cwd() / "calibration_report.json"
    report = [
        {
            "metric": ci.metric,
            "point_estimate": ci.point_estimate,
            "lower": ci.lower,
            "upper": ci.upper,
            "confidence_level": ci.confidence_level,
            "n_labeled": ci.n_labeled,
            "n_total": ci.n_total,
        }
        for ci in intervals
    ]
    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"[dim]Wrote calibration report to {output_path}[/dim]")


@app.command()
def version() -> None:
    """Print the installed raggov version."""
    console.print(f"raggov {_package_version()}")


@app.command("parser-profile-init")
def parser_profile_init(
    name: str = typer.Option("custom", help="Name of the generated parser-validation profile."),
    parser_name: str = typer.Option("", help="Parser or pipeline name."),
    chunking_strategy: str = typer.Option("unknown", help="Declared chunking strategy."),
    page_field: str = typer.Option("", help="Dotted path for page metadata."),
    provenance_field: str = typer.Option("", help="Dotted path for source element provenance."),
    table_field: str = typer.Option("", help="Dotted path for source table provenance."),
    section_field: str = typer.Option("", help="Dotted path for section/header metadata."),
    parent_field: str = typer.Option("", help="Dotted path for parent-child metadata."),
    infer_from_legacy: bool = typer.Option(
        False,
        "--infer-from-legacy/--no-infer-from-legacy",
        help="Allow temporary fallback to legacy metadata aliases during migration.",
    ),
    output: Path = typer.Option(
        Path("parser_validation_profile.yaml"),
        help="Where to write the generated profile YAML.",
    ),
) -> None:
    """Generate a parser-validation profile YAML without optional questionnaire dependencies."""
    yaml_text = profile_yaml_from_answers(
        {
            "name": name,
            "parser_name": parser_name,
            "chunking_strategy": chunking_strategy,
            "page_field": page_field,
            "provenance_field": provenance_field,
            "table_field": table_field,
            "section_field": section_field,
            "parent_field": parent_field,
            "infer_from_legacy": infer_from_legacy,
        }
    )

    output.write_text(yaml_text, encoding="utf-8")
    console.print(f"[dim]Wrote parser-validation profile to {output}[/dim]")


@app.command("stresslab-suite")
def stresslab_suite(
    profile: str = typer.Option("lan", help="Stresslab runtime profile name."),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Run suite without external model calls."),
    case_ids: list[str] = typer.Option(
        None,
        "--case-id",
        help="Specific curated case IDs to run. Defaults to all cases.",
    ),
    output_json: Path = typer.Option(
        Path("stresslab_suite_report.json"),
        help="Where to write the JSON summary artifact.",
    ),
    output_md: Path = typer.Option(
        Path("stresslab_suite_report.md"),
        help="Where to write the markdown summary artifact.",
    ),
    min_match_rate: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="Fail the command if the suite match rate falls below this threshold.",
    ),
) -> None:
    """Run curated stresslab cases and emit suite-level reports."""
    selected_case_ids = case_ids or list_stresslab_cases()
    result = run_suite(selected_case_ids, profile=profile, dry_run=dry_run)

    write_suite_report(result, output_json)
    write_suite_markdown_report(result, output_md)

    console.print(_stresslab_panel(result, profile=profile, dry_run=dry_run))
    console.print(f"[dim]Wrote JSON report to {output_json}[/dim]")
    console.print(f"[dim]Wrote Markdown report to {output_md}[/dim]")

    if result.match_rate < min_match_rate:
        console.print(
            f"[red]Stresslab suite match rate {result.match_rate:.0%} "
            f"is below required threshold {min_match_rate:.0%}[/red]"
        )
        raise typer.Exit(code=1)


@app.command("stresslab-diagnosis")
def stresslab_diagnosis(
    case_ids: list[str] = typer.Option(
        None,
        "--case-id",
        help="Specific diagnosis-golden case IDs to run. Defaults to all cases.",
    ),
    output_json: Path = typer.Option(
        Path("stresslab_diagnosis_report.json"),
        help="Where to write the JSON summary artifact.",
    ),
    output_md: Path = typer.Option(
        Path("stresslab_diagnosis_report.md"),
        help="Where to write the markdown summary artifact.",
    ),
    min_match_rate: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="Fail the command if the diagnosis-golden match rate falls below this threshold.",
    ),
) -> None:
    """Run diagnosis-native golden cases and emit suite-level reports."""
    selected_case_ids = case_ids or list_diagnosis_golden_cases()
    result = run_diagnosis_suite(selected_case_ids)

    write_diagnosis_suite_report(result, output_json)
    write_diagnosis_suite_markdown_report(result, output_md)

    console.print(_diagnosis_suite_panel(result))
    console.print(f"[dim]Wrote JSON report to {output_json}[/dim]")
    console.print(f"[dim]Wrote Markdown report to {output_md}[/dim]")

    if result.match_rate < min_match_rate:
        console.print(
            f"[red]Diagnosis-golden match rate {result.match_rate:.0%} "
            f"is below required threshold {min_match_rate:.0%}[/red]"
        )
        raise typer.Exit(code=1)


@app.command("stresslab-claim-diagnosis")
def stresslab_claim_diagnosis(
    gold_set: Path = typer.Option(
        Path("stresslab/cases/golden/claim_diagnosis_gold_v1.json"),
        help="Path to claim-level gold set JSON.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/claim_diagnosis"),
        help="Directory for claim-diagnosis report artifacts.",
    ),
    enable_a2p: bool = typer.Option(
        True,
        "--enable-a2p/--no-enable-a2p",
        help="Compatibility flag. Harness currently runs with A2P enabled.",
    ),
    enable_a2p_v2: bool = typer.Option(
        False,
        "--enable-a2p-v2/--no-enable-a2p-v2",
        help="Run the existing A2P v2 attribution mode through the claim-diagnosis harness.",
    ),
    report_format: str = typer.Option(
        "both",
        "--format",
        help="Report format: json, markdown, or both.",
    ),
) -> None:
    """Run claim-level diagnosis harness and emit JSON/Markdown reports."""
    if enable_a2p_v2 and not enable_a2p:
        console.print("[red]Invalid options[/red]: --enable-a2p-v2 requires A2P to be enabled")
        raise typer.Exit(code=1)

    format_value = report_format.lower()
    if format_value not in {"json", "markdown", "both"}:
        console.print("[red]Invalid format[/red]: use json, markdown, or both")
        raise typer.Exit(code=1)

    result = run_claim_diagnosis_suite(
        gold_set,
        engine_config={
            "enable_a2p": enable_a2p,
            "use_llm": False,
            "use_a2p_v2": enable_a2p_v2,
        },
    )
    json_path = output_dir / "claim_diagnosis_report.json"
    md_path = output_dir / "claim_diagnosis_report.md"

    if format_value in {"json", "both"}:
        write_claim_diagnosis_report(result, json_path)
        console.print(f"[dim]Wrote JSON report to {json_path}[/dim]")
    if format_value in {"markdown", "both"}:
        write_claim_diagnosis_markdown_report(result, md_path)
        console.print(f"[dim]Wrote Markdown report to {md_path}[/dim]")

    console.print(_claim_diagnosis_panel(result))


def _load_run(run_file: Path) -> RAGRun:
    with run_file.open() as file:
        payload = json.load(file)
    return RAGRun.model_validate(payload)


def _print_validation_error(exc: Exception) -> None:
    if isinstance(exc, ValidationError):
        console.print("[red]Invalid RAGRun[/red]")
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            console.print(f"- [bold]{location}[/bold]: {error['msg']}")
        return
    console.print(f"[red]Invalid RAGRun[/red]: {exc}")


def _calibration_panel(intervals: list[ConfidenceInterval], n_samples: int) -> Panel:
    """Create a Rich Panel displaying ARES calibration results."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Point Estimate", justify="right")
    table.add_column("Confidence Interval", justify="center")
    table.add_column("Width", justify="right")

    for ci in intervals:
        width = ci.upper - ci.lower
        metric_display = ci.metric.replace("_", " ").title()
        table.add_row(
            metric_display,
            f"{ci.point_estimate:.3f}",
            f"[{ci.lower:.3f}, {ci.upper:.3f}]",
            f"{width:.3f}",
        )

    confidence_pct = f"{intervals[0].confidence_level:.0%}"
    title = f"ARES Calibration Report (n={n_samples}, confidence={confidence_pct})"

    return Panel(table, title=title, expand=False)


def _diagnosis_panel(diagnosis: Diagnosis) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Run ID", diagnosis.run_id)
    table.add_row("Timestamp", diagnosis.created_at.isoformat())
    table.add_row("Mode", diagnosis.diagnosis_mode)
    if diagnosis.external_signals_used:
        table.add_row("External Signals", ", ".join(diagnosis.external_signals_used))
    if diagnosis.external_adapter_errors:
        table.add_row("Adapter Errors", ", ".join(diagnosis.external_adapter_errors), style="red")
    table.add_row("Primary failure", _failure_text(diagnosis.primary_failure))
    table.add_row(
        "Should have answered",
        "Yes" if diagnosis.should_have_answered else "No",
    )
    table.add_row("Security risk", _security_risk_text(diagnosis.security_risk))
    if diagnosis.confidence is not None:
        table.add_row("Confidence", f"{diagnosis.confidence:.2f}")

    evidence = "\n".join(f"• {item}" for item in diagnosis.evidence) or "• None"
    table.add_row("Evidence", evidence)
    table.add_row("Recommended fix", Text(diagnosis.recommended_fix, style="bold"))
    table.add_row("Checks", _checks_table(diagnosis))

    return Panel(table, title=f"Diagnosis: {diagnosis.run_id}", expand=False)


def _stresslab_panel(result: Any, *, profile: str, dry_run: bool) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Profile", profile)
    table.add_row("Mode", "dry-run" if dry_run else "live")
    table.add_row("Cases", str(result.total_count))
    table.add_row("Matched", str(result.matched_count))
    table.add_row("Match rate", f"{result.match_rate:.0%}")
    mismatches = ", ".join(result.mismatched_case_ids or []) or "None"
    table.add_row("Mismatches", mismatches)

    return Panel(table, title="Stresslab Suite", expand=False)


def _diagnosis_suite_panel(result: Any) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Cases", str(result.total_count))
    table.add_row("Matched", str(result.matched_count))
    table.add_row("Match rate", f"{result.match_rate:.0%}")
    mismatches = ", ".join(result.mismatched_case_ids or []) or "None"
    table.add_row("Mismatches", mismatches)

    return Panel(table, title="Diagnosis Golden Suite", expand=False)


def _claim_diagnosis_panel(result: Any) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Evaluation status", str(result.evaluation_status))
    table.add_row("A2P mode", str(result.a2p_mode))
    table.add_row("Cases", str(result.total_examples))
    table.add_row("Claim label accuracy", f"{result.claim_label_accuracy:.0%}")
    table.add_row("Sufficiency accuracy", f"{result.sufficiency_accuracy:.0%}")
    table.add_row("A2P cause accuracy", f"{result.a2p_primary_cause_accuracy:.0%}")
    table.add_row("Primary stage accuracy", f"{result.primary_stage_accuracy:.0%}")
    table.add_row("Fix partial accuracy", f"{result.fix_category_partial_accuracy:.0%}")
    table.add_row("Mismatches", str(len(result.mismatches)))

    return Panel(table, title="Claim Diagnosis Harness", expand=False)


def _checks_table(diagnosis: Diagnosis) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Failure")

    for result in diagnosis.analyzer_results:
        table.add_row(
            result.analyzer_name,
            _status_text(result.status),
            result.failure_type.value if result.failure_type is not None else "",
        )

    return table


def _failure_text(failure_type: FailureType) -> Text:
    if failure_type == FailureType.CLEAN:
        return Text(failure_type.value, style="green")
    return Text(failure_type.value, style="red")


def _security_risk_text(security_risk: SecurityRisk) -> Text:
    styles = {
        SecurityRisk.NONE: "green",
        SecurityRisk.LOW: "yellow",
        SecurityRisk.MEDIUM: "orange3",
        SecurityRisk.HIGH: "red",
    }
    return Text(security_risk.value, style=styles[security_risk])


def _status_text(status: str) -> Text:
    styles = {
        "pass": "green",
        "warn": "yellow",
        "fail": "red",
        "skip": "dim",
    }
    return Text(status, style=styles.get(status, "white"))


def _package_version() -> str:
    try:
        return metadata.version("raggov")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def main() -> Any:
    """Run the Typer application."""
    return app()
