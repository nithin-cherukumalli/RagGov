#!/usr/bin/env python3
"""
GovRAG-Calib Dataset Readiness Evaluator.

Does NOT run GovRAG analyzers.
Evaluates dataset readiness only.

Reports:
  - total cases
  - complete real cases (non-placeholder, split≠unset)
  - placeholder cases
  - cases by category, domain, label source, split
  - train/dev/heldout availability
  - minimum per-category count
  - calibration readiness status:
      NOT_READY            if complete_cases < 30
      PROVISIONAL_DATASET  if 30 <= complete_cases < 150
      READY_FOR_HELDOUT_EVAL if complete_cases >= 150 and heldout exists
  - production_gating_eligible = false  (always — no production gating yet)

Outputs:
  - reports/govrag_calib_readiness.md
  - reports/govrag_calib_readiness.json

Usage:
  python scripts/evaluate_govrag_calib.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

def _add_project_to_path() -> None:
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    evals = PROJECT_ROOT / "evals"
    if str(evals) not in sys.path:
        sys.path.insert(0, str(evals))


_add_project_to_path()

from govrag_calib.schema import GovRAGCalibCase  # noqa: E402


# ---------------------------------------------------------------------------
# Category mapping (mirrors validate_govrag_calib.py)
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "CLEAN": "clean_pass",
    "RETRIEVAL_ANOMALY": "retrieval",
    "RETRIEVAL_DEPTH_LIMIT": "retrieval",
    "SCOPE_VIOLATION": "retrieval",
    "STALE_RETRIEVAL": "retrieval",
    "INCONSISTENT_CHUNKS": "retrieval",
    "EMBEDDING_DRIFT": "retrieval",
    "RERANKER_FAILURE": "retrieval",
    "UNSUPPORTED_CLAIM": "grounding",
    "CONTRADICTED_CLAIM": "grounding",
    "INSUFFICIENT_CONTEXT": "sufficiency",
    "CITATION_MISMATCH": "citation",
    "POST_RATIONALIZED_CITATION": "citation",
    "PRIVACY_VIOLATION": "security",
    "PROMPT_INJECTION": "security",
    "SUSPICIOUS_CHUNK": "security",
    "LOW_CONFIDENCE": "confidence",
    "GENERATION_IGNORE": "answer_quality",
    "TABLE_STRUCTURE_LOSS": "parsing",
    "HIERARCHY_FLATTENING": "parsing",
    "METADATA_LOSS": "parsing",
    "PARSER_STRUCTURE_LOSS": "parsing",
    "CHUNKING_BOUNDARY_ERROR": "parsing",
    "INCOMPLETE_DIAGNOSIS": "other",
}

# Minimum recommended count per category for a provisional dataset
MIN_PER_CATEGORY = 3


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list[GovRAGCalibCase]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(GovRAGCalibCase(**data))
            except Exception:
                pass
    return cases


def evaluate_readiness(path: Path) -> dict[str, Any]:
    all_cases = load_cases(path)

    # Partition
    complete_cases = [
        c for c in all_cases
        if c.label_confidence != "low" and c.split != "unset"
    ]
    placeholder_cases = [c for c in all_cases if c.label_confidence == "low"]

    # Counts by dimension
    by_category: Counter[str] = Counter()
    for c in complete_cases:
        cat = CATEGORY_MAP.get(c.expected_primary_failure, "other")
        by_category[cat] += 1

    by_domain: Counter[str] = Counter(c.domain for c in complete_cases)
    by_label_source: Counter[str] = Counter(c.label_source for c in complete_cases)
    by_split: Counter[str] = Counter(c.split for c in complete_cases)
    by_confidence: Counter[str] = Counter(c.label_confidence for c in all_cases)

    # Split availability
    train_count = by_split.get("train", 0)
    dev_count = by_split.get("dev", 0)
    heldout_count = by_split.get("heldout", 0)

    heldout_available = heldout_count > 0
    train_available = train_count > 0
    dev_available = dev_count > 0

    # Per-category minimum check
    thin_categories = {cat: cnt for cat, cnt in by_category.items() if cnt < MIN_PER_CATEGORY}

    # Readiness status
    n_complete = len(complete_cases)
    if n_complete < 30:
        readiness_status = "NOT_READY"
        readiness_reason = (
            f"Only {n_complete} complete cases. Minimum 30 required for any calibration use."
        )
    elif n_complete < 150:
        readiness_status = "PROVISIONAL_DATASET"
        readiness_reason = (
            f"{n_complete} complete cases. Provisional dataset — usable for early "
            f"harness development but not for production calibration evaluation."
        )
    else:
        if heldout_available:
            readiness_status = "READY_FOR_HELDOUT_EVAL"
            readiness_reason = (
                f"{n_complete} complete cases with heldout split available. "
                f"Dataset is ready for held-out calibration evaluation."
            )
        else:
            readiness_status = "PROVISIONAL_DATASET"
            readiness_reason = (
                f"{n_complete} complete cases but no heldout split defined. "
                f"Assign heldout cases before running final calibration evaluation."
            )

    # Production gating — always false
    production_gating_eligible = False
    production_gating_reason = (
        "Production gating is not enabled. No calibrated confidence scores exist. "
        "Activate only after statistical calibration is complete and validated."
    )

    return {
        "dataset_path": str(path),
        "total_cases": len(all_cases),
        "complete_cases": n_complete,
        "placeholder_cases": len(placeholder_cases),
        "by_category": dict(by_category),
        "by_domain": dict(by_domain),
        "by_label_source": dict(by_label_source),
        "by_split": dict(by_split),
        "by_confidence": dict(by_confidence),
        "split_availability": {
            "train": train_available,
            "dev": dev_available,
            "heldout": heldout_available,
            "train_count": train_count,
            "dev_count": dev_count,
            "heldout_count": heldout_count,
        },
        "thin_categories": thin_categories,
        "min_per_category_threshold": MIN_PER_CATEGORY,
        "readiness_status": readiness_status,
        "readiness_reason": readiness_reason,
        "production_gating_eligible": production_gating_eligible,
        "production_gating_reason": production_gating_reason,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

READINESS_ICONS = {
    "NOT_READY": "🔴",
    "PROVISIONAL_DATASET": "🟡",
    "READY_FOR_HELDOUT_EVAL": "🟢",
}


def write_json_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON report → {path}")


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    icon = READINESS_ICONS.get(result["readiness_status"], "⚪")
    lines = [
        "# GovRAG-Calib Readiness Report",
        "",
        f"**Readiness Status:** {icon} `{result['readiness_status']}`",
        "",
        f"> {result['readiness_reason']}",
        "",
        "## Dataset Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total cases | {result['total_cases']} |",
        f"| Complete cases | {result['complete_cases']} |",
        f"| Placeholder cases (low confidence) | {result['placeholder_cases']} |",
        "",
        "## Cases by Category",
        "",
        "| Category | Complete Cases |",
        "|---|---|",
    ]
    for cat, count in sorted(result["by_category"].items(), key=lambda x: -x[1]):
        thin = " ⚠️" if cat in result["thin_categories"] else ""
        lines.append(f"| {cat} | {count}{thin} |")

    lines += [
        "",
        "## Cases by Domain",
        "",
        "| Domain | Count |",
        "|---|---|",
    ]
    for domain, count in sorted(result["by_domain"].items(), key=lambda x: -x[1]):
        lines.append(f"| {domain} | {count} |")

    lines += [
        "",
        "## Cases by Label Source",
        "",
        "| Label Source | Count |",
        "|---|---|",
    ]
    for src, count in sorted(result["by_label_source"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {count} |")

    lines += [
        "",
        "## Split Availability",
        "",
        "| Split | Available | Count |",
        "|---|---|---|",
    ]
    sa = result["split_availability"]
    lines.append(f"| train | {'✅' if sa['train'] else '❌'} | {sa['train_count']} |")
    lines.append(f"| dev | {'✅' if sa['dev'] else '❌'} | {sa['dev_count']} |")
    lines.append(f"| heldout | {'✅' if sa['heldout'] else '❌'} | {sa['heldout_count']} |")

    if result["thin_categories"]:
        lines += [
            "",
            "## Thin Categories (below minimum threshold)",
            "",
            f"Minimum recommended: {result['min_per_category_threshold']} cases per category.",
            "",
        ]
        for cat, count in result["thin_categories"].items():
            lines.append(f"- ⚠️ `{cat}`: {count} case(s)")

    lines += [
        "",
        "## Production Gating",
        "",
        f"**production_gating_eligible:** `{result['production_gating_eligible']}`",
        "",
        f"> {result['production_gating_reason']}",
        "",
        "---",
        "",
        "_This report is generated by `scripts/evaluate_govrag_calib.py`. "
        "No analyzer logic was executed. "
        "No calibrated confidence scores were used._",
    ]

    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Markdown report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GovRAG-Calib dataset readiness.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    path: Path = args.dataset
    if not path.exists():
        print(f"ERROR: Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating readiness: {path}")
    result = evaluate_readiness(path)

    print(f"\nResults:")
    print(f"  Total cases:      {result['total_cases']}")
    print(f"  Complete cases:   {result['complete_cases']}")
    print(f"  Placeholder:      {result['placeholder_cases']}")
    print(f"  Readiness:        {result['readiness_status']}")
    print(f"  Prod gating:      {result['production_gating_eligible']}")

    write_json_report(result, REPORTS_DIR / "govrag_calib_readiness.json")
    write_markdown_report(result, REPORTS_DIR / "govrag_calib_readiness.md")

    print(f"\n  {READINESS_ICONS.get(result['readiness_status'], '⚪')} {result['readiness_status']}")
    print(f"  {result['readiness_reason']}")


if __name__ == "__main__":
    main()
