#!/usr/bin/env python3
"""Taxonomy support check — keeps the failure-type enum honest about its data.

P1 of the foundation stabilization plan. Every `FailureType` is classified by how
many *real* (non-placeholder) golden cases back it in the locked dataset:

    supported   >= 5 real cases
    thin        1-4 real cases
    unsupported 0  real cases  (in the enum but undetectable/unvalidatable)

This guards against the project advertising failure modes it cannot actually
diagnose. It compares the live dataset against the committed
`taxonomy_support_tiers.json` and fails on drift.

Usage:
    python scripts/check_taxonomy_support.py
    python scripts/check_taxonomy_support.py --regenerate   # after dataset grows
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
TIERS = ROOT / "evals" / "govrag_calib" / "taxonomy_support_tiers.json"
MANIFEST = ROOT / "evals" / "govrag_calib" / "DATASET_MANIFEST.json"
FLOOR = 5


def _tier(n: int) -> str:
    return "supported" if n >= FLOOR else ("thin" if n >= 1 else "unsupported")


def _failure_types() -> list[str]:
    """Parse FailureType member values from source (no heavy package import)."""
    import re

    src = (ROOT / "src" / "raggov" / "models" / "diagnosis.py").read_text()
    block = re.search(r"class FailureType\b.*?(?=\nclass )", src, re.S)
    body = block.group(0) if block else src
    return re.findall(r'^\s+[A-Z_]+ = "([A-Z_]+)"', body, re.M)


def _build() -> dict:
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    placeholders = set(json.loads(MANIFEST.read_text())["placeholder_case_ids"])
    real = Counter(
        r["expected_primary_failure"] for r in rows if r["case_id"] not in placeholders
    )
    allc = Counter(r["expected_primary_failure"] for r in rows)
    types = {
        ft: {
            "real_cases": real.get(ft, 0),
            "file_cases": allc.get(ft, 0),
            "tier": _tier(real.get(ft, 0)),
        }
        for ft in _failure_types()
    }
    return {
        "generated": json.loads(MANIFEST.read_text())["frozen_at"],
        "dataset_version": json.loads(MANIFEST.read_text())["dataset_version"],
        "floor_for_supported": FLOOR,
        "definition": {
            "supported": ">=5 real cases",
            "thin": "1-4 real cases",
            "unsupported": "0 real cases (in enum but undetectable/unvalidatable)",
        },
        "types": dict(sorted(types.items(), key=lambda x: (-x[1]["real_cases"], x[0]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    built = _build()
    if args.regenerate:
        TIERS.write_text(json.dumps(built, indent=2) + "\n")
        print("taxonomy support: regenerated taxonomy_support_tiers.json")
        return 0

    committed = json.loads(TIERS.read_text())
    if committed.get("types") == built["types"]:
        tc = Counter(v["tier"] for v in built["types"].values())
        print(
            "taxonomy support: PASS "
            f"(supported={tc['supported']}, thin={tc['thin']}, "
            f"unsupported={tc['unsupported']} of {len(built['types'])} types)"
        )
        return 0

    print("taxonomy support: FAIL — tier file is stale vs the dataset.")
    for t, info in built["types"].items():
        old = committed.get("types", {}).get(t, {})
        if old.get("tier") != info["tier"] or old.get("real_cases") != info["real_cases"]:
            print(f"  {t}: committed={old.get('tier')}({old.get('real_cases')}) "
                  f"live={info['tier']}({info['real_cases']})")
    print("  Run --regenerate after an approved dataset change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
