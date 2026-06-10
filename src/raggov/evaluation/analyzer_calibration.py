"""Audit-only analyzer-wise reliability summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType


class AnalyzerCalibrationCase(BaseModel):
    """Minimal per-case input for analyzer-wise audit metrics."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    case_id: str
    category: str
    mode: str
    expected_primary: str
    expected_stage: str
    actual_primary: str
    actual_stage: str
    selected_analyzer: str | None = None
    analyzer_results: list[AnalyzerResult] = Field(default_factory=list)


@dataclass
class _AnalyzerAccumulator:
    analyzer_name: str
    candidate_count: int = 0
    selected_count: int = 0
    true_positive_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0
    stage_correct_count: int = 0
    failure_type_correct_count: int = 0
    false_clean_contribution: int = 0
    evidence_strength_distribution: Counter[str] = field(default_factory=Counter)
    method_status_distribution: Counter[str] = field(default_factory=Counter)
    calibration_status_distribution: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        precision = (
            self.true_positive_count / self.selected_count
            if self.selected_count
            else None
        )
        recall_denominator = self.true_positive_count + self.false_negative_count
        recall = (
            self.true_positive_count / recall_denominator
            if recall_denominator
            else None
        )
        stage_accuracy = (
            self.stage_correct_count / self.selected_count
            if self.selected_count
            else None
        )
        failure_type_accuracy = (
            self.failure_type_correct_count / self.selected_count
            if self.selected_count
            else None
        )
        return {
            "analyzer_name": self.analyzer_name,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "precision": precision,
            "recall": recall,
            "stage_accuracy": stage_accuracy,
            "failure_type_accuracy": failure_type_accuracy,
            "false_clean_contribution": self.false_clean_contribution,
            "evidence_strength_distribution": dict(sorted(self.evidence_strength_distribution.items())),
            "method_status_distribution": dict(sorted(self.method_status_distribution.items())),
            "calibration_status_distribution": dict(sorted(self.calibration_status_distribution.items())),
        }


def compute_analyzer_calibration_metrics(
    cases: list[AnalyzerCalibrationCase],
) -> dict[str, Any]:
    """Compute analyzer-wise audit metrics from benchmark diagnoses.

    This is intentionally descriptive. It does not estimate calibrated
    confidence and does not change diagnosis behavior.
    """

    analyzers: dict[str, _AnalyzerAccumulator] = {}
    totals = {
        "case_count": len(cases),
        "passed_count": 0,
        "failed_count": 0,
        "false_clean_count": 0,
        "false_security_count": 0,
        "false_incomplete_count": 0,
        "unclaimed_expected_count": 0,
    }
    modes: dict[str, dict[str, int]] = defaultdict(lambda: {"case_count": 0, "passed_count": 0})
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"case_count": 0, "passed_count": 0})

    for case in cases:
        passed = (
            case.expected_primary == case.actual_primary
            and case.expected_stage == case.actual_stage
        )
        totals["passed_count" if passed else "failed_count"] += 1
        modes[case.mode]["case_count"] += 1
        categories[case.category]["case_count"] += 1
        if passed:
            modes[case.mode]["passed_count"] += 1
            categories[case.category]["passed_count"] += 1

        if case.actual_primary == FailureType.CLEAN.value and case.expected_primary != FailureType.CLEAN.value:
            totals["false_clean_count"] += 1
        if case.actual_stage == FailureStage.SECURITY.value and case.expected_stage != FailureStage.SECURITY.value:
            totals["false_security_count"] += 1
        if case.actual_primary == FailureType.INCOMPLETE_DIAGNOSIS.value and case.expected_primary != FailureType.INCOMPLETE_DIAGNOSIS.value:
            totals["false_incomplete_count"] += 1

        selected_result = _selected_result(case)
        selected_analyzer = case.selected_analyzer or (selected_result.analyzer_name if selected_result else None)
        exact_expected_emitters: set[str] = set()

        for result in case.analyzer_results:
            if result.status not in {"fail", "warn"} or result.failure_type is None:
                continue
            acc = analyzers.setdefault(
                result.analyzer_name,
                _AnalyzerAccumulator(result.analyzer_name),
            )
            acc.candidate_count += 1
            _add_signal_distributions(acc, result)

            result_primary = result.failure_type.value
            result_stage = result.stage.value if result.stage is not None else None
            if result_primary == case.expected_primary and result_stage == case.expected_stage:
                exact_expected_emitters.add(result.analyzer_name)

        if selected_result is not None and selected_analyzer is not None:
            acc = analyzers.setdefault(
                selected_analyzer,
                _AnalyzerAccumulator(selected_analyzer),
            )
            acc.selected_count += 1
            selected_primary = (
                selected_result.failure_type.value
                if selected_result.failure_type is not None
                else case.actual_primary
            )
            selected_stage = (
                selected_result.stage.value
                if selected_result.stage is not None
                else case.actual_stage
            )
            if selected_primary == case.expected_primary:
                acc.failure_type_correct_count += 1
            if selected_stage == case.expected_stage:
                acc.stage_correct_count += 1
            if selected_primary == case.expected_primary and selected_stage == case.expected_stage:
                acc.true_positive_count += 1
            else:
                acc.false_positive_count += 1
                if selected_primary == FailureType.CLEAN.value and case.expected_primary != FailureType.CLEAN.value:
                    acc.false_clean_contribution += 1

        if not passed:
            for analyzer_name in exact_expected_emitters:
                if analyzer_name != selected_analyzer:
                    analyzers[analyzer_name].false_negative_count += 1
            if not exact_expected_emitters:
                totals["unclaimed_expected_count"] += 1

    analyzer_rows = [acc.as_dict() for acc in analyzers.values()]
    analyzer_rows.sort(
        key=lambda row: (
            -(row["selected_count"] or 0),
            -(row["candidate_count"] or 0),
            row["analyzer_name"],
        )
    )
    readiness_gaps = _readiness_gaps(analyzer_rows)

    return {
        "audit_only": True,
        "calibration_claim_made": False,
        "production_gating_enabled": False,
        "totals": {
            **totals,
            "pass_rate": totals["passed_count"] / totals["case_count"] if totals["case_count"] else 0.0,
        },
        "modes": {
            mode: {
                **stats,
                "pass_rate": stats["passed_count"] / stats["case_count"] if stats["case_count"] else 0.0,
            }
            for mode, stats in sorted(modes.items())
        },
        "categories": {
            category: {
                **stats,
                "pass_rate": stats["passed_count"] / stats["case_count"] if stats["case_count"] else 0.0,
            }
            for category, stats in sorted(categories.items())
        },
        "analyzers": analyzer_rows,
        "readiness_gaps": readiness_gaps,
    }


def render_analyzer_calibration_markdown(payload: dict[str, Any]) -> str:
    """Render analyzer calibration audit metrics as markdown."""

    totals = payload["totals"]
    lines = [
        "# Analyzer-wise Calibration Audit",
        "",
        "This report is audit-only. It does not change runtime diagnosis behavior, "
        "does not enable production gating, and does not claim calibrated confidence.",
        "",
        "## Summary",
        "",
        f"- Cases: {totals['case_count']}",
        f"- Passed: {totals['passed_count']} ({totals['pass_rate']:.1%})",
        f"- Failed: {totals['failed_count']}",
        f"- False CLEAN: {totals['false_clean_count']}",
        f"- False SECURITY: {totals['false_security_count']}",
        f"- False INCOMPLETE: {totals['false_incomplete_count']}",
        f"- Unclaimed expected failures: {totals['unclaimed_expected_count']}",
        "",
        "## Modes",
        "",
        "| Mode | Cases | Passed | Pass Rate |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for mode, stats in payload["modes"].items():
        lines.append(
            f"| {mode} | {stats['case_count']} | {stats['passed_count']} | {stats['pass_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Categories",
            "",
            "| Category | Cases | Passed | Pass Rate |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for category, stats in payload["categories"].items():
        lines.append(
            f"| {category} | {stats['case_count']} | {stats['passed_count']} | {stats['pass_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Analyzer Metrics",
            "",
            "| Analyzer | Candidates | Selected | TP | FP | FN | Precision | Recall | Stage Acc | Type Acc | False CLEAN |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["analyzers"]:
        lines.append(
            "| {analyzer_name} | {candidate_count} | {selected_count} | {true_positive_count} | "
            "{false_positive_count} | {false_negative_count} | {precision} | {recall} | "
            "{stage_accuracy} | {failure_type_accuracy} | {false_clean_contribution} |".format(
                **{
                    **row,
                    "precision": _fmt_metric(row["precision"]),
                    "recall": _fmt_metric(row["recall"]),
                    "stage_accuracy": _fmt_metric(row["stage_accuracy"]),
                    "failure_type_accuracy": _fmt_metric(row["failure_type_accuracy"]),
                }
            )
        )

    lines.extend(
        [
            "",
            "## Analyzer Readiness Gaps",
            "",
        ]
    )
    if payload.get("readiness_gaps"):
        for gap in payload["readiness_gaps"]:
            lines.append(
                f"- `{gap['analyzer_name']}`: {gap['gap_type']} - {gap['reason']}"
            )
    else:
        lines.append("- No analyzer-level readiness gaps detected in this benchmark slice.")

    lines.extend(
        [
            "",
            "## Metadata Distributions",
            "",
        ]
    )
    for row in payload["analyzers"]:
        if not (
            row["evidence_strength_distribution"]
            or row["method_status_distribution"]
            or row["calibration_status_distribution"]
        ):
            continue
        lines.extend(
            [
                f"### {row['analyzer_name']}",
                "",
                f"- Evidence strength: `{row['evidence_strength_distribution']}`",
                f"- Method status: `{row['method_status_distribution']}`",
                f"- Calibration status: `{row['calibration_status_distribution']}`",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _selected_result(case: AnalyzerCalibrationCase) -> AnalyzerResult | None:
    if case.selected_analyzer is not None:
        for result in case.analyzer_results:
            if result.analyzer_name == case.selected_analyzer:
                return result
    for result in case.analyzer_results:
        if (
            result.failure_type is not None
            and result.failure_type.value == case.actual_primary
            and result.stage is not None
            and result.stage.value == case.actual_stage
        ):
            return result
    return None


def _add_signal_distributions(acc: _AnalyzerAccumulator, result: AnalyzerResult) -> None:
    if not result.signal_metadata:
        acc.evidence_strength_distribution["missing"] += 1
        acc.method_status_distribution["missing"] += 1
        acc.calibration_status_distribution["missing"] += 1
        return
    for signal in result.signal_metadata:
        acc.evidence_strength_distribution[str(signal.evidence_strength)] += 1
        acc.method_status_distribution[str(signal.method_status)] += 1
        acc.calibration_status_distribution[str(signal.calibration_status)] += 1


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _readiness_gaps(analyzer_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for row in analyzer_rows:
        analyzer_name = row["analyzer_name"]
        candidate_count = int(row["candidate_count"])
        selected_count = int(row["selected_count"])
        if candidate_count <= 0:
            continue

        calibration_dist = row["calibration_status_distribution"]
        method_dist = row["method_status_distribution"]
        strength_dist = row["evidence_strength_distribution"]

        missing_metadata = max(
            int(calibration_dist.get("missing", 0)),
            int(method_dist.get("missing", 0)),
            int(strength_dist.get("missing", 0)),
        )
        if missing_metadata:
            gaps.append(
                {
                    "analyzer_name": analyzer_name,
                    "gap_type": "missing_signal_metadata",
                    "reason": f"{missing_metadata} candidate signal(s) lack explicit metadata fields.",
                }
            )

        uncalibrated = int(calibration_dist.get("uncalibrated", 0)) + int(calibration_dist.get("unknown", 0))
        if uncalibrated:
            gaps.append(
                {
                    "analyzer_name": analyzer_name,
                    "gap_type": "not_calibrated",
                    "reason": f"{uncalibrated} signal metadata record(s) are uncalibrated or unknown.",
                }
            )

        weak_or_advisory = int(strength_dist.get("weak", 0)) + int(strength_dist.get("advisory", 0))
        if weak_or_advisory:
            gaps.append(
                {
                    "analyzer_name": analyzer_name,
                    "gap_type": "weak_or_advisory_evidence",
                    "reason": f"{weak_or_advisory} signal metadata record(s) are weak or advisory.",
                }
            )

        if selected_count and row["precision"] is not None and row["precision"] < 1.0:
            gaps.append(
                {
                    "analyzer_name": analyzer_name,
                    "gap_type": "selected_false_positive",
                    "reason": "Selected analyzer has false positives in the evaluated benchmark slice.",
                }
            )
    return gaps
