"""
Threshold sweep for the GovRAG HeuristicValueOverlapVerifier.

Sweeps configurable parameters across their feasible ranges, evaluates each
configuration against the claim-grounding gold dataset, and selects a
recommended configuration according to the high-trust RAG selection policy:

    Primary objective:   minimize false_pass_rate
    Secondary objective: maximize macro F1

A threshold selected on synthetic seed cases is NOT production calibration.
Do not automatically promote the sweep result to production defaults.
Set calibration_status = 'dev_calibrated_seed' explicitly to record provenance.

Usage:
    # From repo root:
    python evals/claim_grounding/sweep_thresholds.py

    # Save outputs:
    python evals/claim_grounding/sweep_thresholds.py \\
        --jsonl-out evals/claim_grounding/reports/sweep.jsonl \\
        --md-out evals/claim_grounding/reports/sweep.md

    # Narrow grid for CI:
    python evals/claim_grounding/sweep_thresholds.py --fast
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.run_eval import (  # noqa: E402
    load_dataset,
    predict,
    compute_metrics,
)
from evals.claim_grounding.calibration import (  # noqa: E402
    CalibrationStatus,
    infer_calibration_status,
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector  # noqa: E402
from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier  # noqa: E402
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder  # noqa: E402

_DEFAULT_DATASET = _EVAL_DIR / "seed_cases.jsonl"

# ---------------------------------------------------------------------------
# Sweep grid definition
# ---------------------------------------------------------------------------

# Full grid — used in normal mode
FULL_GRID: dict[str, list[Any]] = {
    "support_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "anchor_weight": [0.4, 0.5, 0.6, 0.7, 0.8],
    "value_match_score_boost": [0.1, 0.2, 0.3],
    "missing_critical_value_behavior": ["unsupported", "contradicted"],
    "candidate_top_k": [1, 3, 5],
}

# Fast grid — minimal coverage for CI or quick smoke tests
FAST_GRID: dict[str, list[Any]] = {
    "support_threshold": [0.4, 0.5, 0.6],
    "anchor_weight": [0.5, 0.6],
    "value_match_score_boost": [0.2],
    "missing_critical_value_behavior": ["unsupported"],
    "candidate_top_k": [3],
}

# Minimum dataset size to allow "dev_calibrated_seed" status
_MIN_CASES_FOR_DEV_CALIBRATION = 20

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Single sweep configuration result."""

    # Config
    support_threshold: float
    anchor_weight: float
    value_match_score_boost: float
    missing_critical_value_behavior: str
    candidate_top_k: int

    # Metrics
    overall_accuracy: float
    false_pass_rate: float
    false_fail_rate: float
    macro_f1: float
    entailed_f1: float
    unsupported_f1: float
    contradicted_f1: float
    evidence_chunk_recall: float
    fallback_rate: float

    # Selection
    selection_score: float = 0.0   # lower is better (primary: false_pass, secondary: -macro_f1)
    is_recommended: bool = False
    calibration_status: str = "uncalibrated"
    selection_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def selection_score(result: SweepResult) -> tuple[float, float]:
    """
    GovRAG high-trust selection policy.

    Returns a (primary, secondary) tuple where lower is better.
    Primary:   false_pass_rate  — minimize at all costs
    Secondary: -macro_f1        — maximize macro F1 as tiebreaker

    Configurations with the same false_pass_rate are ranked by macro_f1.
    Configurations that increase macro F1 by allowing more false passes are
    NEVER selected — the primary objective is hard.
    """
    return (result.false_pass_rate, -result.macro_f1)


def _macro_f1(metrics: dict[str, Any]) -> float:
    lm = metrics.get("label_metrics", {})
    f1s = [
        lm.get(label, {}).get("f1", 0.0)
        for label in ("entailed", "unsupported", "contradicted")
    ]
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sweep(
    dataset_path: Path = _DEFAULT_DATASET,
    grid: dict[str, list[Any]] | None = None,
    jsonl_out: Path | None = None,
    md_out: Path | None = None,
) -> list[SweepResult]:
    """Run the full threshold sweep and return results."""
    cases = load_dataset(dataset_path)
    n_cases = len(cases)
    print(f"Loaded {n_cases} cases from {dataset_path.name}")

    cal_status = infer_calibration_status(n_cases, dataset_path.name)

    active_grid = grid or FULL_GRID
    param_names = list(active_grid.keys())
    param_values = list(active_grid.values())

    total_configs = 1
    for v in param_values:
        total_configs *= len(v)
    print(f"Sweeping {total_configs} configurations …")

    results: list[SweepResult] = []

    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))

        verifier = HeuristicValueOverlapVerifier(config)
        selector = EvidenceCandidateSelector(config)
        builder = ClaimEvidenceBuilder(verifier, selector)

        predictions = [predict(case, builder) for case in cases]
        metrics = compute_metrics(cases, predictions)

        mf1 = _macro_f1(metrics)
        lm = metrics.get("label_metrics", {})

        result = SweepResult(
            support_threshold=config["support_threshold"],
            anchor_weight=config["anchor_weight"],
            value_match_score_boost=config["value_match_score_boost"],
            missing_critical_value_behavior=config["missing_critical_value_behavior"],
            candidate_top_k=config["candidate_top_k"],
            overall_accuracy=metrics.get("overall_accuracy", 0.0),
            false_pass_rate=metrics.get("false_pass_rate", 1.0),
            false_fail_rate=metrics.get("false_fail_rate", 1.0),
            macro_f1=mf1,
            entailed_f1=lm.get("entailed", {}).get("f1", 0.0),
            unsupported_f1=lm.get("unsupported", {}).get("f1", 0.0),
            contradicted_f1=lm.get("contradicted", {}).get("f1", 0.0),
            evidence_chunk_recall=metrics.get("evidence_chunk_recall", 0.0),
            fallback_rate=metrics.get("fallback_rate", 0.0),
            calibration_status=cal_status,
        )
        results.append(result)

    # ---- Select recommended config ----------------------------------------
    results.sort(key=selection_score)
    best = results[0]
    best.is_recommended = True
    best.selection_notes = [
        f"Recommended by high-trust selection policy: minimize false_pass_rate={best.false_pass_rate:.4f}, "
        f"then maximize macro_F1={best.macro_f1:.4f}.",
        f"Calibration status: {best.calibration_status}.",
        "A threshold selected on synthetic seed cases is NOT production calibration.",
        "To promote to production, run against a blind annotated production dataset.",
    ]

    print(f"\n✓ Sweep complete. {len(results)} configurations evaluated.")
    print(f"  Recommended: support_threshold={best.support_threshold}, "
          f"anchor_weight={best.anchor_weight}, top_k={best.candidate_top_k}")
    print(f"  false_pass_rate={best.false_pass_rate:.4f}, macro_F1={best.macro_f1:.4f}")

    # ---- Outputs ------------------------------------------------------------
    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
        print(f"\nJSONL written to: {jsonl_out}")

    if md_out is not None:
        md = render_sweep_markdown(results, best, dataset_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(md, encoding="utf-8")
        print(f"Markdown written to: {md_out}")

    return results


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_sweep_markdown(
    results: list[SweepResult],
    recommended: SweepResult,
    dataset_path: Path,
) -> str:
    top_n = results[:10]  # top 10 by selection score

    lines = [
        "# Claim-Grounding Threshold Sweep Report",
        "",
        f"- **Dataset**: `{dataset_path.name}`  ({len(results)} configs evaluated)",
        f"- **Selection policy**: minimize `false_pass_rate`, then maximize macro F1",
        "",
        "> **Note**: A threshold selected on synthetic seed cases is NOT production calibration.",
        "> Set `calibration_status = 'dev_calibrated_seed'` to record provenance.",
        "> Do not automatically force sweep results into production defaults.",
        "",
        "## Recommended Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| `support_threshold` | `{recommended.support_threshold}` |",
        f"| `anchor_weight` | `{recommended.anchor_weight}` |",
        f"| `value_match_score_boost` | `{recommended.value_match_score_boost}` |",
        f"| `missing_critical_value_behavior` | `{recommended.missing_critical_value_behavior}` |",
        f"| `candidate_top_k` | `{recommended.candidate_top_k}` |",
        f"| **false_pass_rate** | **`{recommended.false_pass_rate:.4f}`** |",
        f"| false_fail_rate | `{recommended.false_fail_rate:.4f}` |",
        f"| macro F1 | `{recommended.macro_f1:.4f}` |",
        f"| overall accuracy | `{recommended.overall_accuracy:.4f}` |",
        f"| calibration_status | `{recommended.calibration_status}` |",
        "",
    ]
    for note in recommended.selection_notes:
        lines.append(f"> {note}")
    lines.append("")

    lines += [
        "## Top 10 Configurations (ranked by selection policy)",
        "",
        "| rank | sup_thr | anc_w | vm_boost | miss_crit | top_k | fpr | ffr | macro_f1 | acc |",
        "|------|---------|-------|----------|-----------|-------|-----|-----|----------|-----|",
    ]
    for i, r in enumerate(top_n, start=1):
        marker = " ★" if r.is_recommended else ""
        lines.append(
            f"| {i}{marker} "
            f"| {r.support_threshold} "
            f"| {r.anchor_weight} "
            f"| {r.value_match_score_boost} "
            f"| {r.missing_critical_value_behavior[:4]} "
            f"| {r.candidate_top_k} "
            f"| {r.false_pass_rate:.3f} "
            f"| {r.false_fail_rate:.3f} "
            f"| {r.macro_f1:.3f} "
            f"| {r.overall_accuracy:.3f} |"
        )

    # Extremes summary
    worst_fpr = max(results, key=lambda r: r.false_pass_rate)
    best_macro = max(results, key=lambda r: r.macro_f1)
    lines += [
        "",
        "## Extremes",
        "",
        f"- Highest false_pass_rate: `{worst_fpr.false_pass_rate:.4f}` "
        f"(sup={worst_fpr.support_threshold}, anc={worst_fpr.anchor_weight}, "
        f"top_k={worst_fpr.candidate_top_k})",
        f"- Best macro F1: `{best_macro.macro_f1:.4f}` "
        f"(sup={best_macro.support_threshold}, anc={best_macro.anchor_weight}, "
        f"top_k={best_macro.candidate_top_k}, fpr={best_macro.false_pass_rate:.4f})",
        "",
        "## Production Promotion Checklist",
        "",
        "Before promoting any sweep result to production defaults:",
        "- [ ] Run on a blind production dataset (not seed_cases.jsonl)",
        "- [ ] Confirm `false_pass_rate < 0.10` on real queries",
        "- [ ] Get sign-off from the product team on false_fail_rate trade-off",
        "- [ ] Update `calibration_status` to `production_calibrated`",
        "- [ ] Record dataset name, date, and annotator in the calibration config",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GovRAG claim-grounding threshold sweep."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to JSONL evaluation dataset.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller grid for quick smoke tests.",
    )
    p.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Write JSONL sweep results to this path.",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Write markdown report to this path.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_sweep(
        dataset_path=args.dataset,
        grid=FAST_GRID if args.fast else FULL_GRID,
        jsonl_out=args.jsonl_out,
        md_out=args.md_out,
    )


if __name__ == "__main__":
    main()
