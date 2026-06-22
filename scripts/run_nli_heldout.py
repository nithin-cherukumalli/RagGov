"""Run the hybrid entailment (NLI) tier on the real heldout and report the lift.

Compares NATIVE (heuristic) vs LLM-ENTAILMENT on the real heldout, focusing on the #1
trust metric: CLEAN false-positive rate. Requires a reachable LLM (GROQ_API_KEY) — run
on a machine with open network (the build sandbox proxy-blocks api.groq.com).

    GROQ_API_KEY=... PYTHONPATH=src:. python scripts/run_nli_heldout.py

Honest contract: with no/blocked LLM, llm_entailment silently falls back to the heuristic
verifier (both arms identical) — that is reported, not hidden.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raggov_score import build_run, _load_rows  # noqa: E402
from raggov.engine import DiagnosisEngine  # noqa: E402

HELDOUT = (
    Path(__file__).resolve().parent.parent
    / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
)


def _score(rows, engine):
    n = correct = clean_total = clean_fp = 0
    total_claims = fallback_claims = 0
    by = {}
    for case in rows:
        exp = case.get("expected_primary_failure")
        if not exp:
            continue
        diag = engine.diagnose(build_run(case))
        got = diag.primary_failure.value
        n += 1
        correct += got == exp
        by.setdefault(exp, [0, 0])
        by[exp][0] += 1
        by[exp][1] += got == exp
        if exp == "CLEAN":
            clean_total += 1
            clean_fp += got != "CLEAN"
        for c in diag.claim_results:
            total_claims += 1
            if c.fallback_used:
                fallback_claims += 1
    fallback_pct = (fallback_claims / total_claims * 100) if total_claims else 0.0
    return {
        "overall": f"{correct}/{n} = {correct/n:.4f}" if n else "n/a",
        "clean_fp_rate": f"{clean_fp}/{clean_total} = {clean_fp/clean_total:.4f}" if clean_total else "n/a",
        "fallback_pct": f"{fallback_claims}/{total_claims} = {fallback_pct:.2f}%",
        "per_type": {t: f"{v[1]}/{v[0]}" for t, v in by.items()},
    }


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "kimi", "mock", "local_nli"],
        help="local_nli = offline CrossEncoder NLI (no rate limit, no fallback) — preferred for "
        "measurement; cloud providers (groq/kimi) hit rate limits at ~239 sequential calls.",
    )
    ap.add_argument("--model", default=None, help="model id (provider default if omitted)")
    ap.add_argument("--device", default=None, help="local_nli device: cpu | cuda | mps")
    ap.add_argument("--mock", action="store_true", help="offline wiring test (no API call)")
    args = ap.parse_args()
    rows = _load_rows(HELDOUT)

    native = DiagnosisEngine()
    print("NATIVE (heuristic):", _score(rows, native))

    # local_nli runs the offline CrossEncoder verifier inside the engine — no llm_client, no
    # network at inference (model is downloaded once, then cached), so no rate-limit fallback.
    if args.provider == "local_nli":
        config = {"claim_grounding_verifier_policy": "local_nli"}
        if args.model:
            config["nli_model_name"] = args.model
        if args.device:
            config["nli_device"] = args.device
        nli = DiagnosisEngine(config=config)
        print("LOCAL-NLI:        ", _score(rows, nli))
        return

    if args.mock or args.provider == "mock":
        class _MockEntailmentClient:
            def chat(self, prompt: str) -> str:
                return '{"label":"entailed","rationale":"mock"}'
        client = _MockEntailmentClient()
    elif args.provider == "kimi":
        from kimi_client import KimiClient
        client = KimiClient(model=args.model)
    else:
        from groq_client import GroqClient
        client = GroqClient(model=args.model) if args.model else GroqClient()

    nli = DiagnosisEngine(config={
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    })
    print("LLM-ENTAILMENT:   ", _score(rows, nli))


if __name__ == "__main__":
    main()
