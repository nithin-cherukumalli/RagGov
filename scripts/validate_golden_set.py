"""Phase 1A: Baseline Validation Runner

IMPORTANT: This script runs diagnosis validation against AVAILABLE fixtures only.

Current Status:
- Golden set v1 defines 48 expected test cases
- Only ~10 diagnosis fixtures currently exist with complete RAGRun data
- This runner validates against available fixtures and flags coverage gaps

This is Phase 1A baseline validation, NOT calibration.
Sample size is insufficient for threshold calibration (requires 150-300 samples).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raggov import diagnose_file
from raggov.models.diagnosis import FailureStage, FailureType

from stresslab.cases import list_diagnosis_golden_cases, load_diagnosis_golden_case
from stresslab.diagnosis_evaluation import evaluate_diagnosis_case


@dataclass
class AnalyzerConfusionMatrix:
    """Per-analyzer confusion metrics."""

    analyzer_name: str
    total_cases: int
    true_positives: int = 0  # Correctly flagged failure
    false_positives: int = 0  # Flagged when shouldn't have
    true_negatives: int = 0  # Correctly passed
    false_negatives: int = 0  # Missed a failure

    @property
    def precision(self) -> float | None:
        """Precision: TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else None

    @property
    def recall(self) -> float | None:
        """Recall: TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else None

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / Total"""
        return (self.true_positives + self.true_negatives) / self.total_cases if self.total_cases > 0 else 0.0


@dataclass
class BaselineValidationResult:
    """Raw baseline validation results without derived thresholds."""

    total_cases: int
    cases_run: int
    cases_matched: int
    coverage_gap: int  # Number of golden set items without fixtures

    # Per-failure-type metrics
    failure_type_accuracy: dict[str, float]
    failure_stage_accuracy: dict[str, float]

    # Per-analyzer confusion
    analyzer_confusion: dict[str, AnalyzerConfusionMatrix]

    # Mismatches for manual inspection
    mismatched_cases: list[dict[str, Any]]
    unstable_analyzers: list[str]  # Analyzers with <60% agreement

    # Metadata
    run_timestamp: str
    notes: list[str]


def run_baseline_validation() -> BaselineValidationResult:
    """Run baseline validation against available diagnosis fixtures.

    Returns raw metrics without derived thresholds or calibration.
    """
    print("Phase 1A: Baseline Validation")
    print("=" * 60)
    print("WARNING: Running against available fixtures only (~10 cases)")
    print("This is NOT calibration - sample size too small for threshold tuning")
    print()

    # Load available diagnosis golden cases
    case_ids = list_diagnosis_golden_cases()
    print(f"Found {len(case_ids)} diagnosis golden cases with fixtures:")
    for case_id in case_ids:
        print(f"  - {case_id}")
    print()

    # Track metrics
    matched_count = 0
    mismatched_cases: list[dict[str, Any]] = []
    failure_type_matches: Counter[str] = Counter()
    failure_type_total: Counter[str] = Counter()
    failure_stage_matches: Counter[str] = Counter()
    failure_stage_total: Counter[str] = Counter()
    analyzer_results: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0
    })

    # Run diagnosis on each case
    for case_id in case_ids:
        print(f"Running: {case_id}")
        try:
            case = load_diagnosis_golden_case(case_id)
            diagnosis = diagnose_file(case.run_fixture, config=case.engine_config)
            evaluation = evaluate_diagnosis_case(case, diagnosis)

            # Track overall match
            if evaluation.matched_overall:
                matched_count += 1
                print(f"  ✓ MATCHED")
            else:
                print(f"  ✗ MISMATCH")
                mismatched_cases.append({
                    "case_id": case_id,
                    "expected_primary": case.expected_primary_failure,
                    "observed_primary": diagnosis.primary_failure.value,
                    "expected_stage": case.expected_root_cause_stage,
                    "observed_stage": diagnosis.root_cause_stage.value,
                    "notes": evaluation.notes,
                })

            # Track per-failure-type accuracy
            failure_type_total[case.expected_primary_failure] += 1
            if evaluation.matched_primary:
                failure_type_matches[case.expected_primary_failure] += 1

            # Track per-stage accuracy
            failure_stage_total[case.expected_root_cause_stage] += 1
            if evaluation.matched_stage:
                failure_stage_matches[case.expected_root_cause_stage] += 1

            # Track per-analyzer confusion (simplified - would need more detail in real impl)
            # For now, track based on primary failure attribution
            for result in diagnosis.analyzer_results:
                analyzer_name = result.analyzer_name
                analyzer_results[analyzer_name]["total"] += 1

                # Simplified TP/FP logic: if analyzer failed and diagnosis matched, it's TP
                if result.status == "fail":
                    if evaluation.matched_primary:
                        analyzer_results[analyzer_name]["tp"] += 1
                    else:
                        analyzer_results[analyzer_name]["fp"] += 1
                elif result.status == "pass":
                    if case.expected_primary_failure == "CLEAN":
                        analyzer_results[analyzer_name]["tn"] += 1
                    elif not evaluation.matched_primary:
                        analyzer_results[analyzer_name]["fn"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print()
    print("=" * 60)
    print()

    # Calculate per-failure-type accuracy
    failure_type_accuracy = {
        ft: (failure_type_matches[ft] / failure_type_total[ft]) if failure_type_total[ft] > 0 else 0.0
        for ft in failure_type_total
    }

    # Calculate per-stage accuracy
    failure_stage_accuracy = {
        stage: (failure_stage_matches[stage] / failure_stage_total[stage]) if failure_stage_total[stage] > 0 else 0.0
        for stage in failure_stage_total
    }

    # Build analyzer confusion matrices
    analyzer_confusion = {}
    unstable_analyzers = []

    for analyzer_name, metrics in analyzer_results.items():
        cm = AnalyzerConfusionMatrix(
            analyzer_name=analyzer_name,
            total_cases=metrics["total"],
            true_positives=metrics["tp"],
            false_positives=metrics["fp"],
            true_negatives=metrics["tn"],
            false_negatives=metrics["fn"],
        )
        analyzer_confusion[analyzer_name] = cm

        # Flag unstable analyzers (accuracy <60%)
        if cm.accuracy < 0.6:
            unstable_analyzers.append(analyzer_name)

    # Calculate coverage gap
    # Golden set v1 has 48 items, we only ran len(case_ids)
    coverage_gap = 48 - len(case_ids)

    notes = [
        f"Sample size ({len(case_ids)}) is INSUFFICIENT for threshold calibration",
        "ARESCalibrator recommends 150-300 samples for serious calibration",
        f"Coverage gap: {coverage_gap} golden set items lack fixtures",
        "All thresholds derived from this run should be marked PROVISIONAL",
    ]

    return BaselineValidationResult(
        total_cases=48,  # Golden set v1 total
        cases_run=len(case_ids),
        cases_matched=matched_count,
        coverage_gap=coverage_gap,
        failure_type_accuracy=failure_type_accuracy,
        failure_stage_accuracy=failure_stage_accuracy,
        analyzer_confusion=analyzer_confusion,
        mismatched_cases=mismatched_cases,
        unstable_analyzers=unstable_analyzers,
        run_timestamp=datetime.now(UTC).isoformat(),
        notes=notes,
    )


def write_baseline_report(result: BaselineValidationResult, output_path: Path) -> None:
    """Write raw baseline validation report as JSON."""
    # Convert dataclasses to dicts
    analyzer_confusion_dict = {
        name: {
            "analyzer_name": cm.analyzer_name,
            "total_cases": cm.total_cases,
            "true_positives": cm.true_positives,
            "false_positives": cm.false_positives,
            "true_negatives": cm.true_negatives,
            "false_negatives": cm.false_negatives,
            "precision": cm.precision,
            "recall": cm.recall,
            "accuracy": cm.accuracy,
        }
        for name, cm in result.analyzer_confusion.items()
    }

    report = {
        "validation_type": "Phase 1A: Baseline Validation (RAW METRICS ONLY)",
        "total_golden_set_items": result.total_cases,
        "cases_with_fixtures": result.cases_run,
        "cases_matched": result.cases_matched,
        "coverage_gap": result.coverage_gap,
        "match_rate": result.cases_matched / result.cases_run if result.cases_run > 0 else 0.0,
        "failure_type_accuracy": result.failure_type_accuracy,
        "failure_stage_accuracy": result.failure_stage_accuracy,
        "analyzer_confusion_matrices": analyzer_confusion_dict,
        "unstable_analyzers": result.unstable_analyzers,
        "mismatched_cases": result.mismatched_cases,
        "run_timestamp": result.run_timestamp,
        "calibration_status": "NOT_CALIBRATED",
        "notes": result.notes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote baseline validation report: {output_path}")


def render_baseline_summary(result: BaselineValidationResult) -> str:
    """Render human-readable summary of baseline validation."""
    lines = [
        "Phase 1A: Baseline Validation Report",
        "=" * 60,
        "",
        "CRITICAL NOTES:",
    ]
    for note in result.notes:
        lines.append(f"  ⚠️  {note}")

    lines.extend([
        "",
        "Coverage:",
        f"  Total golden set items: {result.total_cases}",
        f"  Cases with fixtures: {result.cases_run}",
        f"  Coverage gap: {result.coverage_gap} items",
        "",
        "Results:",
        f"  Matched cases: {result.cases_matched}/{result.cases_run}",
        f"  Match rate: {result.cases_matched / result.cases_run:.1%}" if result.cases_run > 0 else "  Match rate: N/A",
        "",
        "Per-Failure-Type Accuracy:",
    ])

    for ft, acc in sorted(result.failure_type_accuracy.items()):
        lines.append(f"  {ft}: {acc:.1%}")

    lines.extend(["", "Per-Stage Accuracy:"])
    for stage, acc in sorted(result.failure_stage_accuracy.items()):
        lines.append(f"  {stage}: {acc:.1%}")

    lines.extend(["", "Unstable Analyzers (<60% accuracy):"])
    if result.unstable_analyzers:
        for analyzer in result.unstable_analyzers:
            cm = result.analyzer_confusion[analyzer]
            lines.append(f"  {analyzer}: {cm.accuracy:.1%} accuracy")
    else:
        lines.append("  None")

    lines.extend(["", "Mismatched Cases:"])
    if result.mismatched_cases:
        for case in result.mismatched_cases:
            lines.append(f"  {case['case_id']}:")
            lines.append(f"    Expected: {case['expected_primary']} @ {case['expected_stage']}")
            lines.append(f"    Observed: {case['observed_primary']} @ {case['observed_stage']}")
    else:
        lines.append("  None")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_baseline_validation()

    # Write JSON report
    report_path = Path(__file__).parent.parent / "stresslab" / "reports" / "baseline_validation_v1.json"
    write_baseline_report(result, report_path)

    # Print summary
    print()
    print(render_baseline_summary(result))
