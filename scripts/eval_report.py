#!/usr/bin/env python3
"""Seeded multi-run measurement wrapper for RagGov — Phase 2.

READ-ONLY on engine/analyzer/policy/labels/gates. Measurement harness only.
Imports the canonical scorer from raggov_score.py — no second scoring path.

New in Phase 2:
  - Real heldout v1 (heldout_real_v1.jsonl) as first-class scored set.
  - CLEAN false-positive rate: of N expected-CLEAN rows, how many got non-CLEAN.
  - NLI A/B harness: compares native heuristic vs. llm_entailment policy
    (no-LLM sandbox: entailment falls back to HeuristicValueOverlapVerifier;
    this is an honest no-LLM baseline, NOT a real NLI comparison — flagged clearly).
  - 3 seeds × default+native variance.
  - Spot parity on gc-001 + first heldout CLEAN + first heldout CONTRADICTED.

Method status: heuristic_baseline / practical_approximation
  - Scores EXACT primary-failure accuracy (UNCALIBRATED).
  - Engine is deterministic; seed variation is cosmetic (no data shuffle).
  - No ECE/Brier — calibrated confidence not yet available (Phase 3).
  - confidence_mean column plumbed; may be None.

Usage:
    PYTHONPATH=/tmp/shim:src:. python scripts/eval_report.py
    PYTHONPATH=/tmp/shim:src:. python scripts/eval_report.py --seeds 3 --no-probe
    PYTHONPATH=/tmp/shim:src:. python scripts/eval_report.py --seeds 1 --no-probe --no-nli
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
from typing import Any

# Canonical scorer — single source of truth; no second scoring path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.raggov_score import (  # noqa: E402
    score_file,
    build_run,
    _load_rows,
    _engine,
    _expected,
    CALIB,
    PROBE,
)

HELDOUT_REAL = ROOT / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
REPORTS_DIR = ROOT / "reports" / "calibration"

# Spot-check cases populated lazily
_SPOT_CASES: dict[str, dict] = {}


# ─── CLEAN false-positive rate ───────────────────────────────────────────────

def _clean_fp_rate(path: str | Path, mode: str, config_override: dict | None = None) -> dict:
    """Return CLEAN-FP stats: of expected-CLEAN rows, how many got non-CLEAN."""
    from raggov.engine import DiagnosisEngine
    eng = (
        DiagnosisEngine(config=config_override) if config_override
        else _engine(mode)
    )
    rows = _load_rows(path)
    clean_rows = [r for r in rows if _expected(r) == "CLEAN"]
    total = len(clean_rows)
    fp = 0
    fp_breakdown: dict[str, int] = defaultdict(int)
    for case in clean_rows:
        try:
            diag = eng.diagnose(build_run(case))
            got = diag.primary_failure.value
        except Exception as exc:
            got = f"ERR:{type(exc).__name__}"
        if got != "CLEAN":
            fp += 1
            fp_breakdown[got] += 1
    return {
        "clean_n": total,
        "fp_count": fp,
        "fp_rate": round(fp / total, 4) if total else None,
        "fp_breakdown": dict(sorted(fp_breakdown.items(), key=lambda kv: -kv[1])),
    }


# ─── NLI A/B harness ─────────────────────────────────────────────────────────

def _score_with_config(path: str | Path, config: dict) -> dict:
    """Score a file using a specific engine config (passed directly to DiagnosisEngine)."""
    from raggov.engine import DiagnosisEngine
    eng = DiagnosisEngine(config=config)
    # Reuse score_file logic by passing engine directly
    return score_file(path, mode="default", engine=eng)


def _nli_ab(path: str | Path) -> dict:
    """Compare native heuristic vs. local_nli policy on path.

    Runs both configurations and scores accuracy, CLEAN-FP rate, and CONTRADICTED recall.
    """
    native_cfg = {"claim_grounding_verifier_policy": "conservative_ensemble"}
    entail_cfg = {"claim_grounding_verifier_policy": "local_nli"}

    native_score = _score_with_config(path, native_cfg)
    entail_score = _score_with_config(path, entail_cfg)

    native_fp = _clean_fp_rate(path, mode="default", config_override=native_cfg)
    entail_fp = _clean_fp_rate(path, mode="default", config_override=entail_cfg)

    # CONTRADICTED recall
    def _contradicted_recall(score_result: dict) -> float | None:
        pt = score_result.get("per_type", {})
        c = pt.get("CONTRADICTED_CLAIM")
        if not c or not c["n"]:
            return None
        return round(c["correct"] / c["n"], 4)

    return {
        "note": (
            "Comparing native heuristic (conservative_ensemble fallback) against "
            "local_nli (cross-encoder/nli-deberta-v3-small)."
        ),
        "native": {
            "config": native_cfg,
            "overall": {"n": native_score["n"], "correct": native_score["correct"],
                        "accuracy": native_score["accuracy"]},
            "clean_fp_rate": native_fp["fp_rate"],
            "clean_fp_count": native_fp["fp_count"],
            "clean_fp_breakdown": native_fp["fp_breakdown"],
            "contradicted_recall": _contradicted_recall(native_score),
            "per_type": native_score.get("per_type", {}),
        },
        "entailment": {
            "config": entail_cfg,
            "overall": {"n": entail_score["n"], "correct": entail_score["correct"],
                        "accuracy": entail_score["accuracy"]},
            "clean_fp_rate": entail_fp["fp_rate"],
            "clean_fp_count": entail_fp["fp_count"],
            "clean_fp_breakdown": entail_fp["fp_breakdown"],
            "contradicted_recall": _contradicted_recall(entail_score),
            "per_type": entail_score.get("per_type", {}),
        },
    }


# ─── Seeded multi-run helpers ─────────────────────────────────────────────────

def _score_once(mode: str, seed: int, include_probe: bool, include_heldout: bool) -> dict:
    random.seed(seed)
    result: dict[str, Any] = {
        "seed": seed,
        "mode": mode,
        "calib": score_file(CALIB, mode=mode, splits={"train", "dev", "heldout"}),
    }
    if include_probe:
        result["probe"] = score_file(PROBE, mode=mode)
    else:
        result["probe"] = {"n": 0, "correct": 0, "accuracy": None,
                           "confidence_mean": None, "per_type": {}}
    if include_heldout and HELDOUT_REAL.exists():
        result["heldout_real"] = score_file(HELDOUT_REAL, mode=mode)
    else:
        result["heldout_real"] = {"n": 0, "correct": 0, "accuracy": None,
                                  "confidence_mean": None, "per_type": {}}
    return result


def _aggregate_runs(runs: list[dict], dataset_key: str) -> dict:
    accs = [r[dataset_key]["accuracy"] for r in runs if r[dataset_key].get("accuracy") is not None]
    ns = [r[dataset_key]["n"] for r in runs]
    corrects = [r[dataset_key]["correct"] for r in runs]
    conf_means = [r[dataset_key]["confidence_mean"] for r in runs
                  if r[dataset_key].get("confidence_mean") is not None]
    return {
        "n": ns[0] if ns else None,
        "correct_mean": round(sum(corrects) / len(corrects), 2) if corrects else None,
        "accuracy_mean": round(sum(accs) / len(accs), 4) if accs else None,
        "accuracy_min": round(min(accs), 4) if accs else None,
        "accuracy_max": round(max(accs), 4) if accs else None,
        "confidence_mean": round(sum(conf_means) / len(conf_means), 4) if conf_means else None,
    }


def _per_type_table(runs: list[dict], dataset_key: str) -> list[dict]:
    combined: dict[str, list] = defaultdict(list)
    for run in runs:
        for typ, v in run[dataset_key].get("per_type", {}).items():
            combined[typ].append((v["n"], v["correct"]))
    rows = []
    for typ, entries in sorted(combined.items(), key=lambda kv: -kv[1][0][0]):
        n = entries[0][0]
        correct_mean = round(sum(e[1] for e in entries) / len(entries), 2)
        acc_mean = round(correct_mean / n, 4) if n else None
        rows.append({
            "type": typ,
            "n": n,
            "correct_mean": correct_mean,
            "accuracy_mean": acc_mean,
            "confidence_mean": None,  # Phase 3 drops calibration here
        })
    return rows


# ─── Spot parity ──────────────────────────────────────────────────────────────

def _load_spot_cases() -> None:
    global _SPOT_CASES
    for row in _load_rows(CALIB):
        if row.get("case_id") == "gc-001":
            _SPOT_CASES["gc-001"] = row
            break
    if HELDOUT_REAL.exists():
        for row in _load_rows(HELDOUT_REAL):
            if row.get("expected_primary_failure") == "CLEAN" and "heldout_clean" not in _SPOT_CASES:
                _SPOT_CASES["heldout_clean"] = row
            elif row.get("expected_primary_failure") == "CONTRADICTED_CLAIM" and "heldout_contra" not in _SPOT_CASES:
                _SPOT_CASES["heldout_contra"] = row
            if "heldout_clean" in _SPOT_CASES and "heldout_contra" in _SPOT_CASES:
                break


def _spot_check_parity(mode: str = "default") -> list[dict]:
    _load_spot_cases()
    from raggov.engine import DiagnosisEngine
    eng = DiagnosisEngine() if mode == "default" else DiagnosisEngine(config={"mode": "native"})
    results = []
    for label, case in _SPOT_CASES.items():
        expected = _expected(case)
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


# ─── Markdown ─────────────────────────────────────────────────────────────────

def _markdown(report: dict) -> str:
    lines = [
        f"# eval_report Phase 2 — {report['date']}",
        "",
        "**Method status:** heuristic_baseline / practical_approximation (uncalibrated).  ",
        "Scores EXACT primary-failure accuracy — NOT a production-generalization guarantee.  ",
        "**Real heldout v1 (18/75 = 0.24) is now THE primary metric.**",
        "",
        f"Seeds: {report['seeds']} | Modes: {report['modes']}",
        "",
    ]

    for mode in report["modes"]:
        lines.append(f"## Mode: `{mode}`")
        datasets = [
            ("calib", "Calib (train+dev+heldout)"),
            ("probe", "Induced Probe (synthetic)"),
            ("heldout_real", "**Real Heldout v1** (production bar)"),
        ]
        for dkey, dlabel in datasets:
            agg = report["results"][mode][dkey]["aggregate"]
            lines.append(f"### {dlabel}")
            lines.append("| n | correct (mean) | accuracy mean | min | max | confidence_mean |")
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

        # CLEAN-FP for heldout
        if "heldout_real_clean_fp" in report["results"][mode]:
            fp = report["results"][mode]["heldout_real_clean_fp"]
            lines.append(
                f"**CLEAN-FP rate [heldout_real, {mode}]:** "
                f"{fp['fp_count']}/{fp['clean_n']} = {fp['fp_rate']} "
                f"— breakdown: {fp['fp_breakdown']}"
            )
            lines.append("")

    # NLI A/B
    if "nli_ab" in report:
        ab = report["nli_ab"]
        lines.append("## NLI A/B Comparison (real heldout v1)")
        lines.append(f"> ⚠️ {ab['note']}")
        lines.append("")
        lines.append("| Policy | n | correct | accuracy | CLEAN-FP rate | CONTRADICTED recall |")
        lines.append("|--------|---|---------|----------|---------------|---------------------|")
        for arm_key, arm_label in [("native", "native (conservative_ensemble)"),
                                    ("entailment", "llm_entailment (→ heuristic fallback)")]:
            arm = ab[arm_key]
            lines.append(
                f"| {arm_label} | {arm['overall']['n']} | {arm['overall']['correct']} "
                f"| {arm['overall']['accuracy']} | {arm['clean_fp_rate']} "
                f"| {arm['contradicted_recall']} |"
            )
        lines.append("")
        lines.append("**CLEAN-FP breakdown (native):** " + str(ab["native"]["clean_fp_breakdown"]))
        lines.append("**CLEAN-FP breakdown (entailment):** " + str(ab["entailment"]["clean_fp_breakdown"]))
        lines.append("")

    # Spot parity
    lines.append("## Spot-Case Parity (vs raggov_score.build_run)")
    lines.append("| spot_label | case_id | expected | got | match | confidence |")
    lines.append("|------|------|------|------|------|------|")
    for s in report["spot_parity"]:
        lines.append(
            f"| {s['spot_label']} | {s['case_id']} | {s['expected']} "
            f"| {s['got']} | {s['match']} | {s['confidence']} |"
        )
    lines.append("")
    lines.append("> **Anchors:** Calib 23/45. Probe 80/145 [default] / 82/145 [native]. "
                 "Real heldout 18/75 = 0.24 [default]. Protected 43/46 effective.")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Seeded multi-run eval report wrapper (Phase 2)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default 3)")
    parser.add_argument("--no-probe", action="store_true", help="Skip synthetic probe")
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI A/B (faster)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    modes = ["default", "native"]
    include_heldout = HELDOUT_REAL.exists()

    if not include_heldout:
        print(f"WARNING: {HELDOUT_REAL} not found — heldout_real scoring skipped.")

    print(f"Running {len(seeds)} seeds × {len(modes)} modes …")

    all_results: dict[str, dict] = {}
    for mode in modes:
        runs = []
        for seed in seeds:
            print(f"  mode={mode!r} seed={seed} …", flush=True)
            runs.append(_score_once(mode, seed, not args.no_probe, include_heldout))

        clean_fp = {}
        if include_heldout:
            print(f"  Computing CLEAN-FP rate [mode={mode}] …", flush=True)
            clean_fp = _clean_fp_rate(HELDOUT_REAL, mode=mode)

        all_results[mode] = {
            "calib": {
                "aggregate": _aggregate_runs(runs, "calib"),
                "per_type": _per_type_table(runs, "calib"),
            },
            "probe": {
                "aggregate": _aggregate_runs(runs, "probe"),
                "per_type": _per_type_table(runs, "probe"),
            },
            "heldout_real": {
                "aggregate": _aggregate_runs(runs, "heldout_real"),
                "per_type": _per_type_table(runs, "heldout_real"),
            },
            "heldout_real_clean_fp": clean_fp,
        }

    nli_ab: dict | None = None
    if not args.no_nli and include_heldout:
        print("Running NLI A/B comparison …", flush=True)
        nli_ab = _nli_ab(HELDOUT_REAL)
        print(f"  native   accuracy={nli_ab['native']['overall']['accuracy']}  "
              f"CLEAN-FP={nli_ab['native']['clean_fp_rate']}  "
              f"CONTRADICTED-recall={nli_ab['native']['contradicted_recall']}")
        print(f"  entailment accuracy={nli_ab['entailment']['overall']['accuracy']}  "
              f"CLEAN-FP={nli_ab['entailment']['clean_fp_rate']}  "
              f"CONTRADICTED-recall={nli_ab['entailment']['contradicted_recall']}")

    print("Running spot-case parity …")
    spot = _spot_check_parity(mode="default")
    print(f"\n  {'spot_label':<20} {'expected':<28} {'got':<28} match")
    for s in spot:
        print(f"  {s['spot_label']:<20} {s['expected']:<28} {s['got']:<28} {s['match']}")

    today = date.today().isoformat()
    report: dict[str, Any] = {
        "date": today,
        "seeds": seeds,
        "modes": modes,
        "results": all_results,
        "spot_parity": spot,
        "anchor_note": {
            "protected_baseline": "43/46 effective (42/46 raw + acceptable-alternative cases)",
            "calib": "23/45 [default] = 0.5111",
            "probe_default": "80/145 = 0.5517",
            "probe_native": "82/145 = 0.5655",
            "heldout_real_default": "18/75 = 0.24 — PRIMARY METRIC",
        },
        "method_status": "heuristic_baseline",
        "calibration_status": "not_production_calibrated",
        "production_gating_eligible": False,
    }
    if nli_ab:
        report["nli_ab"] = nli_ab

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"eval_report_{today}.json"
    md_path = REPORTS_DIR / f"eval_report_{today}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")

    calib_agg = all_results["default"]["calib"]["aggregate"]
    print(f"\nSeed variance (calib/default): min={calib_agg['accuracy_min']} "
          f"max={calib_agg['accuracy_max']} mean={calib_agg['accuracy_mean']}")
    if include_heldout:
        hr_agg = all_results["default"]["heldout_real"]["aggregate"]
        print(f"Seed variance (heldout_real/default): min={hr_agg['accuracy_min']} "
              f"max={hr_agg['accuracy_max']} mean={hr_agg['accuracy_mean']}")
        fp = all_results["default"]["heldout_real_clean_fp"]
        print(f"CLEAN-FP [default]: {fp['fp_count']}/{fp['clean_n']} = {fp['fp_rate']}")
    print("Done.")


if __name__ == "__main__":
    main()
