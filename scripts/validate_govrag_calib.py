#!/usr/bin/env python3
"""
Validate the GovRAG-Calib dataset (govrag_calib_150.jsonl).

Checks:
  - Schema validity for every case
  - Unique case IDs
  - Valid failure type
  - Valid stage
  - All retrieved chunks have IDs
  - Claim label evidence IDs reference existing chunks
  - Citation label chunk IDs reference existing chunks
  - label_source, label_confidence, split are present
  - Category distribution
  - Domain distribution
  - Placeholder count (cases with label_confidence=low)

Outputs:
  - reports/govrag_calib_validation.md
  - reports/govrag_calib_validation.json

Usage:
  python scripts/validate_govrag_calib.py [--dataset PATH] [--write-splits]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
SPLITS_DIR = PROJECT_ROOT / "evals" / "govrag_calib" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _add_project_to_path() -> None:
    """Add src/ to sys.path so schema can be imported."""
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    evals = PROJECT_ROOT / "evals"
    if str(evals) not in sys.path:
        sys.path.insert(0, str(evals))


_add_project_to_path()

from govrag_calib.schema import GovRAGCalibCase  # noqa: E402 — after path setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "CLEAN": "clean_pass",
    "RETRIEVAL_ANOMALY": "retrieval",
    "RETRIEVAL_DEPTH_LIMIT": "retrieval",
    "SCOPE_VIOLATION": "retrieval",
    "STALE_RETRIEVAL": "retrieval",
    "INCONSISTENT_CHUNKS": "retrieval",
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
    "EMBEDDING_DRIFT": "retrieval",
    "RERANKER_FAILURE": "retrieval",
    "INCOMPLETE_DIAGNOSIS": "other",
}


def load_raw_lines(path: Path) -> list[dict[str, Any]]:
    raw = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append({"line": i, "data": json.loads(line)})
            except json.JSONDecodeError as e:
                raw.append({"line": i, "error": str(e), "data": None})
    return raw


def validate_dataset(path: Path) -> dict[str, Any]:
    """
    Run all validation checks. Returns a structured result dict.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    valid_cases: list[GovRAGCalibCase] = []

    raw_lines = load_raw_lines(path)

    # --- JSON parse errors ---
    for item in raw_lines:
        if item.get("error"):
            errors.append({"line": item["line"], "type": "json_parse_error", "detail": item["error"]})

    # --- Schema validation ---
    all_data = [item["data"] for item in raw_lines if item.get("data") is not None]
    for data in all_data:
        case_id = data.get("case_id", "unknown")
        try:
            case = GovRAGCalibCase(**data)
            valid_cases.append(case)
        except Exception as e:
            errors.append({"case_id": case_id, "type": "schema_error", "detail": str(e)})

    # --- Unique case IDs ---
    id_counter: Counter[str] = Counter(c.case_id for c in valid_cases)
    for case_id, count in id_counter.items():
        if count > 1:
            errors.append({"case_id": case_id, "type": "duplicate_case_id", "detail": f"Appears {count} times."})

    # --- required fields presence (already enforced by Pydantic, but double-check) ---
    for case in valid_cases:
        if not case.label_source:
            errors.append({"case_id": case.case_id, "type": "missing_label_source"})
        if not case.label_confidence:
            errors.append({"case_id": case.case_id, "type": "missing_label_confidence"})
        if not case.split:
            errors.append({"case_id": case.case_id, "type": "missing_split"})

    # --- Chunk ID completeness ---
    for case in valid_cases:
        for chunk in case.retrieved_chunks:
            if not chunk.chunk_id:
                errors.append({"case_id": case.case_id, "type": "empty_chunk_id", "detail": str(chunk)})

    # --- Claim label evidence IDs reference existing chunks ---
    for case in valid_cases:
        chunk_ids = {c.chunk_id for c in case.retrieved_chunks}
        for cl in case.expected_claim_labels:
            for eid in cl.expected_evidence_chunk_ids:
                if eid not in chunk_ids:
                    errors.append({
                        "case_id": case.case_id,
                        "type": "invalid_evidence_chunk_ref",
                        "detail": f"ClaimLabel '{cl.claim_id}' references chunk_id '{eid}' not in retrieved_chunks.",
                    })

    # --- Citation label chunk IDs reference existing chunks ---
    for case in valid_cases:
        chunk_ids = {c.chunk_id for c in case.retrieved_chunks}
        for cit in case.expected_citation_labels:
            if cit.cited_chunk_id is not None and cit.cited_chunk_id not in chunk_ids:
                errors.append({
                    "case_id": case.case_id,
                    "type": "invalid_citation_chunk_ref",
                    "detail": f"CitationLabel '{cit.citation_id}' references cited_chunk_id '{cit.cited_chunk_id}' not in retrieved_chunks.",
                })

    # --- Category distribution ---
    category_counts: Counter[str] = Counter()
    for case in valid_cases:
        cat = CATEGORY_MAP.get(case.expected_primary_failure, "other")
        category_counts[cat] += 1

    # --- Domain distribution ---
    domain_counts: Counter[str] = Counter(c.domain for c in valid_cases)

    # --- Split distribution ---
    split_counts: Counter[str] = Counter(c.split for c in valid_cases)

    # --- Label source distribution ---
    label_source_counts: Counter[str] = Counter(c.label_source for c in valid_cases)

    # --- Label confidence distribution ---
    label_conf_counts: Counter[str] = Counter(c.label_confidence for c in valid_cases)

    # --- Placeholder count ---
    placeholder_count = label_conf_counts.get("low", 0)

    # --- Complete cases (non-placeholder, non-unset) ---
    complete_cases = [
        c for c in valid_cases
        if c.label_confidence != "low" and c.split != "unset"
    ]

    # --- Warnings for thin categories ---
    for cat, count in category_counts.items():
        if count < 5:
            warnings.append(f"Category '{cat}' has only {count} cases (minimum recommended: 5).")

    return {
        "total_lines": len(raw_lines),
        "total_valid_cases": len(valid_cases),
        "complete_cases": len(complete_cases),
        "placeholder_cases": placeholder_count,
        "errors": errors,
        "warnings": warnings,
        "category_distribution": dict(category_counts),
        "domain_distribution": dict(domain_counts),
        "split_distribution": dict(split_counts),
        "label_source_distribution": dict(label_source_counts),
        "label_confidence_distribution": dict(label_conf_counts),
        "passed": len(errors) == 0,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON report → {path}")


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "✅ PASSED" if result["passed"] else "❌ FAILED"
    lines = [
        "# GovRAG-Calib Validation Report",
        "",
        f"**Status:** {status}",
        f"**Total lines:** {result['total_lines']}",
        f"**Valid cases:** {result['total_valid_cases']}",
        f"**Complete cases (non-placeholder, split≠unset):** {result['complete_cases']}",
        f"**Placeholder cases (label_confidence=low):** {result['placeholder_cases']}",
        f"**Errors:** {len(result['errors'])}",
        f"**Warnings:** {len(result['warnings'])}",
        "",
        "## Category Distribution",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for cat, count in sorted(result["category_distribution"].items()):
        lines.append(f"| {cat} | {count} |")
    lines += [
        "",
        "## Domain Distribution",
        "",
        "| Domain | Count |",
        "|---|---|",
    ]
    for domain, count in sorted(result["domain_distribution"].items()):
        lines.append(f"| {domain} | {count} |")
    lines += [
        "",
        "## Split Distribution",
        "",
        "| Split | Count |",
        "|---|---|",
    ]
    for split, count in sorted(result["split_distribution"].items()):
        lines.append(f"| {split} | {count} |")
    lines += [
        "",
        "## Label Confidence Distribution",
        "",
        "| Confidence | Count |",
        "|---|---|",
    ]
    for conf, count in sorted(result["label_confidence_distribution"].items()):
        lines.append(f"| {conf} | {count} |")

    if result["warnings"]:
        lines += ["", "## Warnings", ""]
        for w in result["warnings"]:
            lines.append(f"- ⚠️ {w}")

    if result["errors"]:
        lines += ["", "## Errors", ""]
        for e in result["errors"]:
            lines.append(f"- ❌ `{e}`")
    else:
        lines += ["", "## Errors", "", "None."]

    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Markdown report → {path}")


def write_splits(valid_cases: list[GovRAGCalibCase], splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_groups: dict[str, list[dict]] = defaultdict(list)
    for c in valid_cases:
        split_groups[c.split].append(json.loads(c.model_dump_json()))
    for split_name, cases in split_groups.items():
        out = splits_dir / f"{split_name}.jsonl"
        with out.open("w") as f:
            for case in cases:
                f.write(json.dumps(case) + "\n")
        print(f"  Wrote {len(cases)} cases → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GovRAG-Calib dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--write-splits", action="store_true", help="Write split JSONL files.")
    args = parser.parse_args()

    dataset_path: Path = args.dataset

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating: {dataset_path}")
    result = validate_dataset(dataset_path)

    print(f"\nResults:")
    print(f"  Valid cases:       {result['total_valid_cases']}")
    print(f"  Complete cases:    {result['complete_cases']}")
    print(f"  Placeholder cases: {result['placeholder_cases']}")
    print(f"  Errors:            {len(result['errors'])}")
    print(f"  Warnings:          {len(result['warnings'])}")

    write_json_report(result, REPORTS_DIR / "govrag_calib_validation.json")
    write_markdown_report(result, REPORTS_DIR / "govrag_calib_validation.md")

    if args.write_splits:
        # Re-load valid cases for splitting
        raw = load_raw_lines(dataset_path)
        valid_cases = []
        for item in raw:
            if item.get("data"):
                try:
                    valid_cases.append(GovRAGCalibCase(**item["data"]))
                except Exception:
                    pass
        write_splits(valid_cases, SPLITS_DIR)

    if result["errors"]:
        print(f"\n{'='*60}")
        print("VALIDATION FAILED — see reports/govrag_calib_validation.md")
        print(f"{'='*60}")
        for e in result["errors"][:10]:
            print(f"  ❌ {e}")
        if len(result["errors"]) > 10:
            print(f"  ... and {len(result['errors']) - 10} more errors.")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("VALIDATION PASSED")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
