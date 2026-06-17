#!/usr/bin/env python3
"""Dataset lock check — fails if the canonical calibration dataset drifts.

This is the enforcement mechanism for P0 of the foundation stabilization plan.
It guards against the exact problem that broke the v2 task queue: golden labels
being relabeled/renumbered between sessions without a record.

It compares the live canonical file against `DATASET_MANIFEST.json`:
  - SHA256 of the file must match, OR
  - if content changed intentionally, the set of immutable case IDs must still
    match and the manifest must have been regenerated (sha256 updated).

Exit code 0 = locked/consistent, 1 = drift detected.

Usage:
    python scripts/check_dataset_lock.py
    python scripts/check_dataset_lock.py --regenerate   # after an approved change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals" / "govrag_calib" / "DATASET_MANIFEST.json"


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_manifest(canonical: Path, version: str, frozen_at: str) -> dict:
    raw = canonical.read_bytes()
    rows = _load_rows(canonical)
    placeholders = sorted(
        r["case_id"]
        for r in rows
        if any(
            tok in json.dumps(r).lower() for tok in ("todo:", "placeholder")
        )
    )
    return {
        "dataset_version": version,
        "frozen_at": frozen_at,
        "canonical_file": str(canonical.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "immutable_case_ids": sorted(r["case_id"] for r in rows),
        "splits": dict(Counter(r.get("split") for r in rows)),
        "per_type_counts_all": dict(
            Counter(r["expected_primary_failure"] for r in rows)
        ),
        "placeholder_case_ids": placeholders,
        "scored_rule": (
            "Only split in {train,dev,heldout} is scored; "
            "split=unset (incl. all placeholders) is excluded."
        ),
        "notes": (
            "IDs are immutable. Any change to this file requires a "
            "LABEL_CHANGELOG.md entry and a manifest sha256 bump. "
            "Enforced by scripts/check_dataset_lock.py."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    canonical = ROOT / manifest["canonical_file"]

    if args.regenerate:
        new = _build_manifest(
            canonical, manifest["dataset_version"], manifest["frozen_at"]
        )
        MANIFEST.write_text(json.dumps(new, indent=2) + "\n")
        print("dataset lock: manifest regenerated (sha256 now "
              f"{new['sha256'][:12]}...). Remember to add a LABEL_CHANGELOG entry.")
        return 0

    raw = canonical.read_bytes()
    live_sha = hashlib.sha256(raw).hexdigest()
    live_ids = sorted(r["case_id"] for r in _load_rows(canonical))

    if live_sha == manifest["sha256"]:
        print(f"dataset lock: PASS (sha256 {live_sha[:12]}..., "
              f"{manifest['row_count']} rows, ids immutable)")
        return 0

    print("dataset lock: FAIL — canonical file differs from manifest.")
    print(f"  manifest sha256: {manifest['sha256'][:12]}...")
    print(f"  live     sha256: {live_sha[:12]}...")
    missing = set(manifest["immutable_case_ids"]) - set(live_ids)
    added = set(live_ids) - set(manifest["immutable_case_ids"])
    if missing:
        print(f"  REMOVED/RENUMBERED ids (forbidden): {sorted(missing)}")
    if added:
        print(f"  NEW ids: {sorted(added)}")
    if not missing and not added:
        print("  IDs unchanged but content changed: an approved edit must run "
              "--regenerate AND add a LABEL_CHANGELOG.md entry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
