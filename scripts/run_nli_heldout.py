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
    by = {}
    for case in rows:
        exp = case.get("expected_primary_failure")
        if not exp:
            continue
        got = engine.diagnose(build_run(case)).primary_failure.value
        n += 1
        correct += got == exp
        by.setdefault(exp, [0, 0])
        by[exp][0] += 1
        by[exp][1] += got == exp
        if exp == "CLEAN":
            clean_total += 1
            clean_fp += got != "CLEAN"
    return {
        "overall": f"{correct}/{n} = {correct/n:.4f}" if n else "n/a",
        "clean_fp_rate": f"{clean_fp}/{clean_total} = {clean_fp/clean_total:.4f}" if clean_total else "n/a",
        "per_type": {t: f"{v[1]}/{v[0]}" for t, v in by.items()},
    }


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--mock", action="store_true", help="offline wiring test (no API call)")
    args = ap.parse_args()
    rows = _load_rows(HELDOUT)

    native = DiagnosisEngine()
    print("NATIVE (heuristic):", _score(rows, native))

    if args.mock:
        class _MockEntailmentClient:
            def chat(self, prompt: str) -> str:
                return '{"label":"entailed","rationale":"mock"}'
        client = _MockEntailmentClient()
    else:
        from groq_client import GroqClient
        client = GroqClient(model=args.model)

    nli = DiagnosisEngine(config={
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    })
    print("LLM-ENTAILMENT:   ", _score(rows, nli))


if __name__ == "__main__":
    main()
