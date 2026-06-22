"""Full Kimi NLI A/B on the locked real heldout (75 rows).

Produces:
  1. NATIVE vs LLM-ENTAILMENT overall/per-type/CLEAN-FP table
  2. Per-claim NLI label dump for first 10 CLEAN rows (grounded-clean gate calibration)
  3. Spot parity check on 1 CLEAN case vs raggov_score

READ-ONLY: no engine/policy/labels/gates changes.

Usage:
    PYTHONPATH=src:. python scripts/run_kimi_ab_full.py --model moonshot-v1-8k
    PYTHONPATH=src:. python scripts/run_kimi_ab_full.py --model moonshot-v1-8k --max-rows 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raggov_score import build_run, _load_rows  # noqa: E402
from raggov.engine import DiagnosisEngine  # noqa: E402

HELDOUT = (
    Path(__file__).resolve().parent.parent
    / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
)


def _score_detailed(rows, engine, label: str):
    """Score with timing, per-type breakdown, and CLEAN-FP breakdown."""
    n = correct = clean_total = clean_fp = 0
    by: dict[str, list[int]] = {}
    clean_fp_breakdown: dict[str, int] = {}
    t0 = time.time()
    for case in rows:
        exp = case.get("expected_primary_failure")
        if not exp:
            continue
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
    elapsed = time.time() - t0
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
        "elapsed_s": round(elapsed, 1),
    }


def _per_claim_clean_dump(rows, engine, max_rows: int = 10):
    """For up to max_rows CLEAN rows, dump per-claim NLI labels from ClaimGroundingAnalyzer."""
    clean_rows = [r for r in rows if r.get("expected_primary_failure") == "CLEAN"][:max_rows]
    results = []
    for case in clean_rows:
        row_id = case.get("case_id") or case.get("id") or "unknown"
        try:
            d = engine.diagnose(build_run(case))
        except Exception as exc:
            results.append({"case_id": row_id, "error": str(exc)})
            continue
        primary = d.primary_failure.value
        claim_labels = []
        for r in getattr(d, "analyzer_results", []) or []:
            if r.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            for c in r.claim_results or []:
                claim_labels.append({
                    "label": getattr(c, "label", "?"),
                    "verification_method": getattr(c, "verification_method", None),
                    "fallback_used": getattr(c, "fallback_used", None),
                    "claim_text": (getattr(c, "claim_text", None) or "")[:80],
                })
        # Tally
        label_counts: dict[str, int] = {}
        for cl in claim_labels:
            lbl = cl["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        results.append({
            "case_id": row_id,
            "primary_got": primary,
            "is_correct_clean": primary == "CLEAN",
            "n_claims": len(claim_labels),
            "label_counts": label_counts,
            "claims": claim_labels,
        })
    return results


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="kimi", choices=["groq", "kimi", "mock"])
    ap.add_argument("--model", default="moonshot-v1-8k")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limit rows for faster runs (None = all 75)")
    ap.add_argument("--clean-dump-n", type=int, default=10,
                    help="Number of CLEAN rows for per-claim dump")
    args = ap.parse_args()

    rows = _load_rows(HELDOUT)
    if args.max_rows:
        rows = rows[: args.max_rows]
        print(f"[NOTE] Running on first {args.max_rows} rows (not full 75)")

    total_rows = len(rows)
    print(f"Loaded {total_rows} rows from {HELDOUT.name}")
    print()

    # --- ARM 1: NATIVE ---
    native_engine = DiagnosisEngine()
    print("=== ARM 1: NATIVE (heuristic) ===")
    native_result = _score_detailed(rows, native_engine, "NATIVE")
    _print_result(native_result)

    # --- ARM 2: LLM-ENTAILMENT (Kimi) ---
    if args.provider == "mock":
        class _MockClient:
            def chat(self, prompt: str) -> str:
                return '{"label":"entailed","rationale":"mock"}'
        client = _MockClient()
    elif args.provider == "kimi":
        from kimi_client import KimiClient
        print(f"\nInitialising Kimi client: model={args.model}")
        client = KimiClient(model=args.model)
        # Quick connectivity test
        try:
            ping = client.chat("Reply with exactly: ok")[:20]
            print(f"Kimi ping OK: {ping!r}")
        except Exception as exc:
            print(f"Kimi ping FAILED: {exc}")
            sys.exit(1)
    else:
        from groq_client import GroqClient
        client = GroqClient(model=args.model) if args.model else GroqClient()

    nli_engine = DiagnosisEngine(config={
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    })

    print(f"\n=== ARM 2: LLM-ENTAILMENT ({args.provider}/{args.model}) ===")
    print(f"[This will make many API calls for {total_rows} rows — please wait...]")
    nli_result = _score_detailed(rows, nli_engine, f"LLM-ENTAILMENT({args.model})")
    _print_result(nli_result)

    # --- A/B COMPARISON TABLE ---
    print("\n=== A/B Comparison Table ===")
    print(f"{'Policy':<40} {'n':>4} {'correct':>7} {'accuracy':>8} {'CLEAN-FP rate':>14} {'elapsed':>8}")
    print("-" * 85)
    for res in [native_result, nli_result]:
        fp_str = f"{res['clean_fp']}/{res['clean_total']}={res['clean_fp_rate']}" if res['clean_fp_rate'] is not None else "n/a"
        print(f"{res['label']:<40} {res['n']:>4} {res['correct']:>7} {res['accuracy']:>8.4f} {fp_str:>14} {res['elapsed_s']:>7}s")

    # --- PER-TYPE COMPARISON ---
    print("\n=== Per-Type Comparison ===")
    all_types = sorted(set(list(native_result["per_type"].keys()) + list(nli_result["per_type"].keys())))
    print(f"{'Type':<30} {'Native n/correct':>18} {'NLI n/correct':>15}")
    print("-" * 65)
    for t in all_types:
        n_v = native_result["per_type"].get(t, {"n": 0, "correct": 0})
        l_v = nli_result["per_type"].get(t, {"n": 0, "correct": 0})
        native_str = f"{n_v['correct']}/{n_v['n']}"
        nli_str = f"{l_v['correct']}/{l_v['n']}"
        print(f"{t:<30} {native_str:>18} {nli_str:>15}")

    # --- PER-CLAIM CLEAN DUMP ---
    print(f"\n=== Per-Claim NLI Labels for first {args.clean_dump_n} CLEAN rows ===")
    print("[Running NLI engine on CLEAN rows for grounded-clean gate calibration]")
    clean_dump = _per_claim_clean_dump(rows, nli_engine, args.clean_dump_n)
    for entry in clean_dump:
        if "error" in entry:
            print(f"  {entry['case_id']}: ERROR {entry['error']}")
            continue
        correct_marker = "✓" if entry["is_correct_clean"] else "✗"
        print(f"  [{correct_marker}] {entry['case_id']}")
        print(f"      primary_got={entry['primary_got']} | n_claims={entry['n_claims']} | label_counts={entry['label_counts']}")
        for cl in entry["claims"][:6]:  # show up to 6 claims
            fb = " [fallback]" if cl.get("fallback_used") else ""
            print(f"        {cl['label']:12} vm={cl.get('verification_method','?')}{fb} | {cl['claim_text']!r}")

    # --- SPOT PARITY CHECK ---
    print("\n=== Spot Parity (raggov_score) ===")
    # Find one CLEAN case
    clean_cases = [r for r in rows if r.get("expected_primary_failure") == "CLEAN"]
    if clean_cases:
        spot = clean_cases[0]
        spot_id = spot.get("case_id") or spot.get("id") or "unknown"
        try:
            native_diag = native_engine.diagnose(build_run(spot))
            nli_diag = nli_engine.diagnose(build_run(spot))
            print(f"  case_id={spot_id}")
            print(f"  expected=CLEAN | native={native_diag.primary_failure.value} | nli={nli_diag.primary_failure.value}")
        except Exception as exc:
            print(f"  spot parity error: {exc}")

    # Return structured for report
    return {
        "native": native_result,
        "nli": nli_result,
        "clean_dump": clean_dump,
    }


def _print_result(res: dict) -> None:
    fp_str = f"{res['clean_fp']}/{res['clean_total']} = {res['clean_fp_rate']}" if res['clean_fp_rate'] is not None else "n/a"
    print(f"  Overall: {res['correct']}/{res['n']} = {res['accuracy']}")
    print(f"  CLEAN-FP: {fp_str}")
    print(f"  Per-type: {res['per_type']}")
    print(f"  CLEAN-FP breakdown: {res['clean_fp_breakdown']}")
    print(f"  Elapsed: {res['elapsed_s']}s")


if __name__ == "__main__":
    main()
