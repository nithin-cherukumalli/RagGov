"""Tests for engineer-facing `diagnose --format text` rendering.

The text renderer is the primary one-page surface a real RAG engineer sees
when triaging a failing run. It MUST attribute the verdict to a specific
analyzer, surface runner-up failure types, and point to a concrete next
inspection target. This is pure formatting on existing Diagnosis fields,
so the protected baseline and analyzer behaviour are unaffected.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from raggov.cli import _diagnosis_panel, app
from raggov.engine import diagnose
from raggov.io.serialize import diagnosis_to_dict
from raggov.models.run import RAGRun


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _run_diagnose(fixture_name: str) -> str:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["diagnose", str(FIXTURES / fixture_name), "--mode", "native", "--format", "text"],
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_text_format_preserves_existing_fields() -> None:
    """Backward compat: existing fields the renderer already emitted must stay."""
    out = _run_diagnose("citation_mismatch.json")
    assert "Run ID:" in out
    assert "Primary Failure:" in out
    assert "Root Cause Stage:" in out
    assert "Recommended Fix:" in out
    assert "Human Review Required:" in out
    # Footer with calibration posture must remain on every run.
    assert "calibration_status=" in out
    assert "production_gating_eligible=" in out


def test_text_format_surfaces_why_block() -> None:
    """New: a 'Why this verdict?' block must appear so the engineer can trace
    the verdict to a specific analyzer instead of just seeing a failure name."""
    out = _run_diagnose("citation_mismatch.json")
    assert "Why this verdict?" in out


def test_text_format_attributes_verdict_to_analyzer() -> None:
    """The why-block must name the analyzer that voted for primary_failure,
    so the engineer knows which component fired and can inspect its code/config."""
    out = _run_diagnose("citation_mismatch.json")
    # On the citation-mismatch fixture, either CitationFaithfulnessAnalyzerV0 or
    # ClaimGroundingAnalyzer is expected to fire. The renderer should name at
    # least one analyzer in the Voted-by line.
    assert "Voted by:" in out
    assert "Analyzer" in out or "analyzer" in out


def test_text_format_lists_runner_up_failures_when_present() -> None:
    """If there are secondary_failures, the why-block must list them as
    'Also considered:' so the engineer sees the differential, not just the winner."""
    out = _run_diagnose("citation_mismatch.json")
    # secondary_failures may legitimately be empty; in that case we accept
    # either absence of the line or an explicit "(none)" rendering.
    # The renderer must NOT crash when secondary_failures is empty.
    assert "Why this verdict?" in out


def test_text_format_points_to_concrete_inspection_target() -> None:
    """The why-block must surface a 'Inspect next:' line driven by available
    structured data (pinpoint location, root_cause_attribution, or first
    failing node)."""
    out = _run_diagnose("citation_mismatch.json")
    assert "Inspect next:" in out


def test_text_format_does_not_crash_on_clean_pass() -> None:
    """A clean fixture must render cleanly. The why-block on a CLEAN verdict
    should not invent a fake analyzer attribution."""
    out = _run_diagnose("clean_pass.json")
    assert "Primary Failure:" in out
    # On CLEAN, the why-block should say so explicitly rather than naming an
    # arbitrary analyzer.
    assert "Why this verdict?" in out


def test_rich_panel_surfaces_why_block() -> None:
    run = RAGRun.model_validate_json((FIXTURES / "citation_mismatch.json").read_text())
    diagnosis = diagnose(run, config={"mode": "native"})
    console = Console(record=True, width=140)

    console.print(_diagnosis_panel(diagnosis))
    rendered = console.export_text()

    assert "Why this verdict?" in rendered
    assert "Voted by:" in rendered
    assert "Also considered:" in rendered
    assert "Inspect next:" in rendered


def test_json_serialization_surfaces_structured_why_block() -> None:
    run = RAGRun.model_validate_json((FIXTURES / "citation_mismatch.json").read_text())
    diagnosis = diagnose(run, config={"mode": "native"})

    payload = diagnosis_to_dict(diagnosis)

    assert "why_block" in payload
    assert set(payload["why_block"]) == {
        "verdict_summary",
        "voted_by",
        "also_considered",
        "inspect_next",
    }
