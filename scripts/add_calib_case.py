#!/usr/bin/env python3
"""Validate and (optionally) append a new calibration case — safely.

Enforces the governance rules from `evals/govrag_calib/DATASET_GOVERNANCE.md` so
new cases can be added without re-breaking the dataset:
  - validates the live format (the one the eval actually consumes)
  - refuses placeholders (no real data) into scored splits
  - assigns the next immutable `gc-0NN` id on --append (never reuses)
  - reminds you to re-lock + log the change

Usage:
    python scripts/add_calib_case.py mycase.json            # validate only
    python scripts/add_calib_case.py mycase.json --append   # validate + add
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
DIAGNOSIS = ROOT / "src" / "raggov" / "models" / "diagnosis.py"

REQUIRED = [
    "domain", "source_type", "query", "retrieved_chunks", "answer", "citations",
    "expected_primary_failure", "expected_stage", "label_source",
    "label_confidence", "split", "notes",
]
VALID_SPLITS = {"train", "dev", "heldout", "unset"}
PLACEHOLDER_TOKENS = ("todo", "placeholder", "fixme")


def _failure_types() -> set[str]:
    src = DIAGNOSIS.read_text()
    block = re.search(r"class FailureType\b.*?(?=\nclass )", src, re.S)
    body = block.group(0) if block else src
    return set(re.findall(r'^\s+[A-Z_]+ = "([A-Z_]+)"', body, re.M))


def _rows() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


def _next_id(rows: list[dict]) -> str:
    nums = [int(m.group(1)) for r in rows
            if (m := re.match(r"gc-(\d+)$", r["case_id"]))]
    return f"gc-{max(nums) + 1:03d}"


def validate(case: dict, existing_ids: set[str], ftypes: set[str]) -> list[str]:
    errs: list[str] = []
    for f in REQUIRED:
        if f not in case:
            errs.append(f"missing required field: {f}")

    cid = case.get("case_id", "")
    if cid in existing_ids:
        errs.append(f"case_id {cid!r} already exists — IDs are immutable, never reuse")

    ept = case.get("expected_primary_failure")
    if ept and ept not in ftypes:
        errs.append(f"expected_primary_failure {ept!r} is not a FailureType enum value")

    split = case.get("split")
    if split and split not in VALID_SPLITS:
        errs.append(f"split {split!r} not in {sorted(VALID_SPLITS)}")

    content = " ".join([
        str(case.get("notes", "")), str(case.get("answer", "")),
        str(case.get("query", "")),
        *(str(c.get("text", "")) for c in (case.get("retrieved_chunks") or [])),
    ]).lower()
    is_placeholder = any(t in content for t in PLACEHOLDER_TOKENS)
    if is_placeholder and split in {"train", "dev", "heldout"}:
        errs.append("placeholder text (TODO/placeholder/…) is not allowed in a scored "
                    "split; keep placeholders in split=unset until filled")

    chunks = case.get("retrieved_chunks") or []
    if not chunks:
        errs.append("retrieved_chunks must have at least one chunk")
    chunk_docs = set()
    for i, c in enumerate(chunks):
        for f in ("chunk_id", "doc_id", "text"):
            if not c.get(f):
                errs.append(f"retrieved_chunks[{i}] missing {f}")
        if c.get("doc_id"):
            chunk_docs.add(c["doc_id"])

    # Soft check: citation doc binding (some failure types intentionally don't bind).
    cits = case.get("citations") or []
    unbound = [d for d in cits if d not in chunk_docs]
    if unbound and ept not in {"CITATION_MISMATCH", "POST_RATIONALIZED_CITATION"}:
        errs.append(f"WARN: citations {unbound} not in retrieved docs {sorted(chunk_docs)} "
                    f"(expected for CITATION_MISMATCH, suspicious otherwise)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_file")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    case = json.loads(Path(args.case_file).read_text())
    rows = _rows()
    ids = {r["case_id"] for r in rows}
    ftypes = _failure_types()

    errs = validate(case, ids, ftypes)
    hard = [e for e in errs if not e.startswith("WARN")]
    for e in errs:
        print(("  ⚠ " if e.startswith("WARN") else "  ✗ ") + e)

    if hard:
        print(f"add_calib_case: FAIL ({len(hard)} error(s)) — not appended.")
        return 1

    if not args.append:
        print("add_calib_case: VALID (run again with --append to add it).")
        return 0

    new_id = _next_id(rows)
    case["case_id"] = new_id
    with DATASET.open("a") as fh:
        fh.write(json.dumps(case) + "\n")
    print(f"add_calib_case: appended as {new_id}.")
    print("NEXT (required):")
    print("  1. python scripts/check_dataset_lock.py --regenerate")
    print(f"  2. add a LABEL_CHANGELOG.md entry for {new_id}")
    print("  3. python scripts/check_taxonomy_support.py --regenerate")
    print("  4. re-run the eval and record accuracy with sample size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
