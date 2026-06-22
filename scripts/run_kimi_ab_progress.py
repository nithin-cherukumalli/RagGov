#!/usr/bin/env python3
"""Run full 75-row Kimi NLI A/B with per-row progress logging.

Writes progress to both stdout and a log file so we can track it.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.disable(logging.CRITICAL)

from raggov_score import build_run, _load_rows
from raggov.engine import DiagnosisEngine
from kimi_client import KimiClient

HELDOUT = Path(__file__).resolve().parent.parent / "evals/govrag_calib/staging/raw/heldout_real_v1.jsonl"
LOG = Path("/tmp/kimi_ab_progress.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def score_with_progress(rows, engine, label):
    n = correct = clean_total = clean_fp = 0
    by = {}
    clean_fp_breakdown = {}
    t0 = time.time()
    for i, case in enumerate(rows):
        exp = case.get("expected_primary_failure")
        if not exp:
            continue
        row_t = time.time()
        try:
            got = engine.diagnose(build_run(case)).primary_failure.value
        except Exception as exc:
            got = f"ERR:{type(exc).__name__}"
        n += 1
        correct += got == exp
        by.setdefault(exp, [0, 0])
        by[exp][0] += 1
        by[exp][1] += got == exp
        if exp == "CLEAN":
            clean_total += 1
            if got != "CLEAN":
                clean_fp += 1
                clean_fp_breakdown[got] = clean_fp_breakdown.get(got, 0) + 1
        row_elapsed = time.time() - row_t
        log(f"  [{label}] row {i:3d}: exp={exp} got={got} ({row_elapsed:.1f}s)")
    total = time.time() - t0
    return {
        "label": label,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "clean_total": clean_total,
        "clean_fp": clean_fp,
        "clean_fp_rate": round(clean_fp / clean_total, 4) if clean_total else None,
        "per_type": {t: {"n": v[0], "correct": v[1]} for t, v in sorted(by.items())},
        "clean_fp_breakdown": dict(sorted(clean_fp_breakdown.items(), key=lambda kv: -kv[1])),
        "elapsed_s": round(total, 1),
    }


LOG.write_text("# Kimi NLI A/B progress log\n")
log(f"Starting. Loaded heldout from {HELDOUT}")
rows = _load_rows(HELDOUT)
log(f"Rows: {len(rows)}")

# ARM 1: NATIVE
log("\n=== ARM 1: NATIVE ===")
native = score_with_progress(rows, DiagnosisEngine(), "NATIVE")
log(f"NATIVE FINAL: {native['correct']}/{native['n']} = {native['accuracy']}, CLEAN-FP: {native['clean_fp']}/{native['clean_total']} = {native['clean_fp_rate']}")
log(f"NATIVE per-type: {native['per_type']}")
log(f"NATIVE clean_fp_breakdown: {native['clean_fp_breakdown']}")

# ARM 2: NLI
log(f"\n=== ARM 2: LLM-ENTAILMENT (kimi/moonshot-v1-8k) ===")
client = KimiClient(model="moonshot-v1-8k")
# Quick ping
ping = client.chat("Reply with exactly: ok")
log(f"Kimi ping: {ping[:20]!r}")
nli = score_with_progress(
    rows,
    DiagnosisEngine(config={"llm_client": client, "claim_grounding_verifier_policy": "llm_entailment"}),
    "NLI"
)
log(f"NLI FINAL: {nli['correct']}/{nli['n']} = {nli['accuracy']}, CLEAN-FP: {nli['clean_fp']}/{nli['clean_total']} = {nli['clean_fp_rate']}")
log(f"NLI per-type: {nli['per_type']}")
log(f"NLI clean_fp_breakdown: {nli['clean_fp_breakdown']}")

# Save JSON
out = {"native": native, "nli": nli, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
Path("/tmp/kimi_ab_results.json").write_text(json.dumps(out, indent=2))
log(f"\nResults saved to /tmp/kimi_ab_results.json")
log("DONE")
