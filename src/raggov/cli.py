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

from raggov.engine import DiagnosisEngine
from raggov.models.diagnosis import Diagnosis, FailureType, SecurityRisk
from raggov.models.run import RAGRun


app = typer.Typer(help="Diagnose retrieval-augmented generation runs.")
console = Console()


@app.command()
def diagnose(run_file: Path) -> None:
    """Diagnose a RAGRun JSON file."""
    try:
        run = _load_run(run_file)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        _print_validation_error(exc)
        raise typer.Exit(code=1) from exc

    diagnosis = DiagnosisEngine().diagnose(run)
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
def version() -> None:
    """Print the installed raggov version."""
    console.print(f"raggov {_package_version()}")


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


def _diagnosis_panel(diagnosis: Diagnosis) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Run ID", diagnosis.run_id)
    table.add_row("Timestamp", diagnosis.created_at.isoformat())
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
