"""Summarize GovRAG-Calib JSONL distribution without running GovRAG.

This utility is read-only. It does not mutate datasets, run analyzers, compute
confidence intervals, or imply production calibration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAMILY_TARGETS = {
    "clean_pass": 15,
    "retrieval": 25,
    "grounding": 25,
    "citation": 20,
    "sufficiency": 15,
    "version_validity": 15,
    "security_privacy": 20,
    "answer_quality": 15,
}


@dataclass(frozen=True)
class CalibSummary:
    path: Path
    total_cases: int
    by_failure_family: Counter[str]
    by_label_status: Counter[str]
    by_calibration_split: Counter[str]
    by_source_suite: Counter[str]
    security_relevant_count: int
    adversarial_count: int
    missing_family_targets: dict[str, int]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: record must be a JSON object")
        records.append(record)
    return records


def summarize_dataset(path: Path) -> CalibSummary:
    records = load_jsonl(path)
    by_failure_family = Counter(str(record.get("failure_family", "<missing>")) for record in records)
    by_label_status = Counter(str(record.get("label_status", "<missing>")) for record in records)
    by_calibration_split = Counter(str(record.get("calibration_split", "<missing>")) for record in records)
    by_source_suite = Counter(str(record.get("source_suite", "<missing>")) for record in records)
    missing_family_targets = {
        family: max(target - by_failure_family.get(family, 0), 0)
        for family, target in FAMILY_TARGETS.items()
    }
    return CalibSummary(
        path=path,
        total_cases=len(records),
        by_failure_family=by_failure_family,
        by_label_status=by_label_status,
        by_calibration_split=by_calibration_split,
        by_source_suite=by_source_suite,
        security_relevant_count=sum(1 for record in records if record.get("security_relevant") is True),
        adversarial_count=sum(1 for record in records if record.get("adversarial") is True),
        missing_family_targets=missing_family_targets,
    )


def _format_counter(counter: Counter[str], *, target_map: dict[str, int] | None = None) -> list[str]:
    keys = list(target_map) if target_map else sorted(counter)
    lines: list[str] = []
    for key in keys:
        count = counter.get(key, 0)
        if target_map:
            lines.append(f"- {key}: {count}/{target_map[key]}")
        else:
            lines.append(f"- {key}: {count}")
    return lines


def format_summary(summary: CalibSummary) -> str:
    lines = [
        f"GovRAG-Calib summary: {summary.path}",
        f"total_cases: {summary.total_cases}",
        "",
        "cases_by_failure_family:",
        *_format_counter(summary.by_failure_family, target_map=FAMILY_TARGETS),
        "",
        "missing_family_targets_to_150:",
    ]
    for family, missing in summary.missing_family_targets.items():
        lines.append(f"- {family}: {missing}")
    lines.extend(
        [
            "",
            "cases_by_label_status:",
            *_format_counter(summary.by_label_status),
            "",
            "cases_by_calibration_split:",
            *_format_counter(summary.by_calibration_split),
            "",
            "cases_by_source_suite:",
            *_format_counter(summary.by_source_suite),
            "",
            f"security_relevant_cases: {summary.security_relevant_count}",
            f"adversarial_cases: {summary.adversarial_count}",
            "",
            "calibration_status: not_calibrated",
            "production_gating_eligible: false",
            "note: summary is read-only and does not run GovRAG or imply production calibration.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    summary = summarize_dataset(args.dataset)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
