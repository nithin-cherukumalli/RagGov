"""Runner for claim-level diagnostic evaluation harness v0."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from stresslab.cases import ClaimDiagnosisGoldSet, load_claim_diagnosis_gold_set
from stresslab.claim_diagnosis_evaluation import (
    ClaimDiagnosisHarnessResult,
    render_claim_diagnosis_report,
    run_claim_diagnosis_harness,
)
from stresslab.reports import write_json_artifact


def run_claim_diagnosis_suite(
    gold_set_path: str | Path,
    engine_config: dict[str, Any] | None = None,
) -> ClaimDiagnosisHarnessResult:
    """Load claim-level gold set and run harness."""
    path = Path(gold_set_path)
    if path.exists():
        gold_set = ClaimDiagnosisGoldSet.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        gold_set = load_claim_diagnosis_gold_set(str(gold_set_path))
    return run_claim_diagnosis_harness(gold_set, engine_config=engine_config)


def write_claim_diagnosis_report(
    result: ClaimDiagnosisHarnessResult,
    output_path: str | Path,
) -> Path:
    """Write claim-level evaluation JSON report."""
    payload = {
        "evaluation_status": result.evaluation_status,
        "a2p_mode": result.a2p_mode,
        "total_examples": result.total_examples,
        "case_count": result.total_examples,
        "aggregate_metrics": {
            "claim_label_accuracy": result.claim_label_accuracy,
            "citation_validity_accuracy": result.citation_validity_accuracy,
            "freshness_validity_accuracy": result.freshness_validity_accuracy,
            "sufficiency_accuracy": result.sufficiency_accuracy,
            "a2p_primary_cause_accuracy": result.a2p_primary_cause_accuracy,
            "primary_stage_accuracy": result.primary_stage_accuracy,
            "fix_category_exact_accuracy": result.fix_category_exact_accuracy,
            "fix_category_partial_accuracy": result.fix_category_partial_accuracy,
            "false_clean_count": result.false_clean_count,
            "claim_label_breakdown": result.claim_label_breakdown,
        },
        "category_metrics": result.category_metrics,
        "per_case_results": [asdict(item) for item in result.per_example],
        "mismatches": result.mismatches,
    }
    return write_json_artifact(output_path, payload)


def write_claim_diagnosis_markdown_report(
    result: ClaimDiagnosisHarnessResult,
    output_path: str | Path,
) -> Path:
    """Write claim-level evaluation markdown report."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_claim_diagnosis_markdown(result)
    target.write_text(markdown, encoding="utf-8")
    return target


def _render_claim_diagnosis_markdown(result: ClaimDiagnosisHarnessResult) -> str:
    lines = [
        "# Claim-Level Diagnostic Evaluation Report",
        "",
        f"- Evaluation status: `{result.evaluation_status}`",
        f"- A2P mode: `{result.a2p_mode}`",
        f"- total_examples: {result.total_examples}",
        f"- Case count: {result.total_examples}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| claim_label_accuracy | {result.claim_label_accuracy:.2f} |",
        f"| citation_validity_accuracy | {result.citation_validity_accuracy:.2f} |",
        f"| freshness_validity_accuracy | {result.freshness_validity_accuracy:.2f} |",
        f"| sufficiency_accuracy | {result.sufficiency_accuracy:.2f} |",
        f"| a2p_primary_cause_accuracy | {result.a2p_primary_cause_accuracy:.2f} |",
        f"| primary_stage_accuracy | {result.primary_stage_accuracy:.2f} |",
        f"| fix_category_exact_accuracy | {result.fix_category_exact_accuracy:.2f} |",
        f"| fix_category_partial_accuracy | {result.fix_category_partial_accuracy:.2f} |",
        f"| false_clean_count | {result.false_clean_count} |",
        f"| claim_label_breakdown | {result.claim_label_breakdown} |",
        "",
        "## Category Metrics",
        "",
        "| Category | Cases | Claim | Sufficiency | Overall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, metrics in sorted(result.category_metrics.items()):
        lines.append(
            f"| {category} | {int(metrics['total_cases'])} | "
            f"{metrics['claim_label_accuracy']:.2f} | "
            f"{metrics['sufficiency_accuracy']:.2f} | "
            f"{metrics['overall_match_rate']:.2f} |"
        )

    lines.extend([
        "",
        "## Per-Case Summary",
    ])
    for case in result.per_example:
        status = "PASS" if case.matched_overall else "FAIL"
        lines.append(f"- `{case.case_id}` ({case.category}): **{status}**")

    lines.extend(["", "## Mismatches"])
    if not result.mismatches:
        lines.append("- None")
    else:
        for mismatch in result.mismatches:
            lines.append(f"- `{mismatch['case_id']}`")
            lines.append(f"  - Expected stage: `{mismatch['expected_primary_stage']}`")
            lines.append(f"  - Observed stage: `{mismatch['observed_primary_stage']}`")
            lines.append(f"  - Expected fix category: `{mismatch['expected_fix_category']}`")
            lines.append(f"  - Observed fix category: `{mismatch['observed_fix_category']}`")
            lines.append(f"  - Notes: {'; '.join(mismatch['notes'])}")

    lines.extend(["", "## Raw Summary", "", "```text", render_claim_diagnosis_report(result), "```", ""])
    return "\n".join(lines)
