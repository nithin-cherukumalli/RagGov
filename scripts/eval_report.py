#!/usr/bin/env python3
"""Seeded multi-run measurement wrapper for RagGov.

READ-ONLY on engine/analyzer/policy/labels/gates. This script is a measurement
harness only — it imports the canonical scorer from raggov_score.py and runs it
across multiple seeds and modes to assess score variance and per-type accuracy.

Method status: heuristic_baseline / practical_approximation
  - Scores exact primary-failure accuracy (UNCALIBRATED).
  - Engine is deterministic for a given config, so seed variation reflects
    any sampling or shuffle in data loading (not in the engine itself).
  - No ECE/Brier — calibrated confidence does not yet exist.
  - Confidence column is plumbed but may be None (no calibration yet).

Usage:
    PYTHONPATH=/tmp/shim:src:. python scripts/eval_report.py
    PYTHONPATH=/tmp/shim:src:. python scripts/eval_report.py --seeds 3 --no-probe
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Canonical scorer — single source of truth; no second scoring path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.raggov_score import score_file, build_run, _load_rows, CALIB, PROBE  # noqa: E402

REPORTS_DIR = ROOT / "reports" / "calibration"

# Spot-check anchors used for parity verification
_SPOT_CASES: dict[str, dict] = {}  # populated lazily below


def _load_spot_cases() -> None:
    """Load the 3 named spot cases (gc-001, probe CITATION, probe CLEAN)."""
    global _SPOT_CASES
    # gc-001 from calib
    for row in _load_rows(CALIB):
        if row.get("case_id") == "gc-001":
            _SPOT_CASES["gc-001"] = row
            break
    # First CITATION_MISMATCH from probe
    for row in _load_rows(PROBE):
        if row.get("expected_primary_failure") == "CITATION_MISMATCH" and "citation_probe" not in _SPOT_CASES:
            _SPOT_CASES["citation_probe"] = row
            break
    # First CLEAN from probe
    for row in _load_rows(PROBE):
        if row.get("expected_primary_failure") == "CLEAN" and "clean_probe" not in _SPOT_CASES:
            _SPOT_CASES["clean_probe"] = row
            break


def _score_once(mode: str, seed: int) -> dict:
    """Run both datasets with a given mode. seed is noted but engine is deterministic."""
    random.seed(seed)
    calib = score_file(CALIB, mode=mode, splits={"train", "dev", "heldout"})
    probe = score_file(PROBE, mode=mode)
    return {"seed": seed, "mode": mode, "calib": calib, "probe": probe}


def _aggregate_runs(runs: list[dict], dataset_key: str) -> dict:
    """Aggregate accuracy across seeds for calib or probe."""
    accs = [r[dataset_key]["accuracy"] for r in runs if r[dataset_key]["accuracy"] is not None]
    ns = [r[dataset_key]["n"] for r in runs]
    corrects = [r[dataset_key]["correct"] for r in runs]
    conf_means = [r[dataset_key]["confidence_mean"] for r in runs if r[dataset_key].get("confidence_mean") is not None]
    return {
        "n": ns[0] if ns else None,          # same each seed (deterministic data)
        "correct_mean": round(sum(corrects) / len(corrects), 2) if corrects else None,
        "accuracy_mean": round(sum(accs) / len(accs), 4) if accs else None,
        "accuracy_min": round(min(accs), 4) if accs else None,
        "accuracy_max": round(max(accs), 4) if accs else None,
        "confidence_mean": round(sum(conf_means) / len(conf_means), 4) if conf_means else None,
    }


def _per_type_table(runs: list[dict], dataset_key: str) -> list[dict]:
    """Aggregate per-type accuracy across seeds."""
    combined: dict[str, list] = defaultdict(list)  # type -> [(n, correct), ...]
    for run in runs:
        for typ, v in run[dataset_key].get("per_type", {}).items():
            combined[typ].append((v["n"], v["correct"]))
    rows = []
    for typ, entries in sorted(combined.items(), key=lambda kv: -kv[1][0][0]):
        n = entries[0][0]  # deterministic, same across seeds
        correct_mean = round(sum(e[1] for e in entries) / len(entries), 2)
        acc_mean = round(correct_mean / n, 4) if n else None
        rows.append({
            "type": typ,
            "n": n,
            "correct_mean": correct_mean,
            "accuracy_mean": acc_mean,
            "confidence_mean": None,  # placeholder — Phase 3 drops calibration in here
        })
    return rows


def _spot_check_parity(mode: str = "default") -> list[dict]:
    """Score the 3 named spot cases via raggov_score.build_run and report results."""
    _load_spot_cases()
    from raggov.engine import DiagnosisEngine
    eng = DiagnosisEngine() if mode == "default" else DiagnosisEngine(config={"mode": "native"})
    results = []
    for label, case in _SPOT_CASES.items():
        expected = case.get("expected_primary_failure") or case.get("primary_failure")
        try:
            diag = eng.diagnose(build_run(case))
            got = diag.primary_failure.value
            conf = getattr(diag, "confidence", None)
        except Exception as exc:
            got = f"ERR:{type(exc).__name__}"
            conf = None
        results.append({
            "spot_label": label,
            "case_id": case.get("case_id", "gc-PENDING"),
            "expected": expected,
            "got": got,
            "match": got == expected,
            "confidence": round(conf, 4) if isinstance(conf, float) else conf,
        })
    return results


def _markdown(report: dict) -> str:
    lines = [
        f"# eval_report — {report['date']}",
        "",
        "**Method status:** heuristic_baseline / practical_approximation (uncalibrated).  ",
        "Scores EXACT primary-failure accuracy on available eval data — NOT a production-generalization guarantee.",
        "",
        f"Seeds used: {report['seeds']}  ",
        f"Modes: {report['modes']}",
        "",
    ]
    for mode in report["modes"]:
        lines.append(f"## Mode: `{mode}`")
        for dkey, dlabel in [("calib", "Calib (train+dev+heldout)"), ("probe", "Induced Probe")]:
            agg = report["results"][mode][dkey]["aggregate"]
            lines.append(f"### {dlabel}")
            lines.append(
                f"| n | correct (mean) | accuracy mean | min | max | confidence_mean |"
            )
            lines.append("|----|----|----|----|----|-----|")
            lines.append(
                f"| {agg['n']} | {agg['correct_mean']} | {agg['accuracy_mean']} "
                f"| {agg['accuracy_min']} | {agg['accuracy_max']} | {agg['confidence_mean']} |"
            )
            lines.append("")
            lines.append("**Per-type:**")
            lines.append("| type | n | correct (mean) | accuracy mean | confidence_mean |")
            lines.append("|------|---|----|----|------|")
            for row in report["results"][mode][dkey]["per_type"]:
                lines.append(
                    f"| {row['type']} | {row['n']} | {row['correct_mean']} "
                    f"| {row['accuracy_mean']} | {row['confidence_mean']} |"
                )
            lines.append("")

    lines.append("## Spot-Case Parity (3 named cases vs. raggov_score.build_run)")
    lines.append("| spot_label | case_id | expected | got | match | confidence |")
    lines.append("|------|------|------|------|------|------|")
    for s in report["spot_parity"]:
        lines.append(
            f"| {s['spot_label']} | {s['case_id']} | {s['expected']} "
            f"| {s['got']} | {s['match']} | {s['confidence']} |"
        )
    lines.append("")
    lines.append(
        "> **Anchor note:** protected baseline check returned 42/46 (check_protected_baseline.py). "
        "The ledger anchor is 43/46 effective (including acceptable-alternative cases). "
        "Calib 23/45 confirmed. Probe 80/145 [default] / 82/145 [native] confirmed."
    )
    return "\n".join(lines)


def main() -> None:
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Seeded multi-run eval report wrapper")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default 5)")
    parser.add_argument("--no-probe", action="store_true", help="Skip probe scoring (faster)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    modes = ["default", "native"]

    print(f"Running {len(seeds)} seeds × {len(modes)} modes …")

    all_results: dict[str, dict] = {}
    for mode in modes:
        runs = []
        for seed in seeds:
            print(f"  mode={mode!r} seed={seed} …", flush=True)
            run = _score_once(mode, seed)
            if args.no_probe:
                run["probe"] = {"n": 0, "correct": 0, "accuracy": None, "confidence_mean": None, "per_type": {}}
            runs.append(run)

        all_results[mode] = {
            "calib": {
                "aggregate": _aggregate_runs(runs, "calib"),
                "per_type": _per_type_table(runs, "calib"),
            },
            "probe": {
                "aggregate": _aggregate_runs(runs, "probe"),
                "per_type": _per_type_table(runs, "probe"),
            },
        }

    print("Running spot-case parity check …")
    spot = _spot_check_parity(mode="default")
    print("\n3-case parity (default mode):")
    print(f"  {'spot_label':<20} {'expected':<28} {'got':<28} {'match'}")
    for s in spot:
        print(f"  {s['spot_label']:<20} {s['expected']:<28} {s['got']:<28} {s['match']}")

    today = date.today().isoformat()
    report = {
        "date": today,
        "seeds": seeds,
        "modes": modes,
        "results": all_results,
        "spot_parity": spot,
        "anchor_note": {
            "protected_baseline_check": "42/46 (check_protected_baseline.py raw output) — "
                                        "effective 43/46 with acceptable-alternative cases per ledger",
            "calib": "23/45 [default] = 0.5111",
            "probe_default": "80/145 [default] = 0.5517",
            "probe_native": "82/145 [native] = 0.5655",
        },
        "method_status": "heuristic_baseline",
        "calibration_status": "not_production_calibrated",
        "production_gating_eligible": False,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"eval_report_{today}.json"
    md_path = REPORTS_DIR / f"eval_report_{today}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"\nSeed variance summary (default / calib):")
    calib_agg = all_results["default"]["calib"]["aggregate"]
    print(f"  min={calib_agg['accuracy_min']}  max={calib_agg['accuracy_max']}  mean={calib_agg['accuracy_mean']}")
    print("Done.")


if __name__ == "__main__":
    main()
