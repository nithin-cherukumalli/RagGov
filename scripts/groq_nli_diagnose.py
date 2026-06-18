"""Diagnose whether the Groq entailment tier actually ENGAGES (vs silent fallback).

Run on a machine with network (sandbox proxy-blocks api.groq.com):

    GROQ_API_KEY=... PYTHONPATH=src:. python scripts/groq_nli_diagnose.py
    GROQ_API_KEY=... PYTHONPATH=src:. python scripts/groq_nli_diagnose.py --model llama-3.1-8b-instant

It prints, in order:
  1) the Groq models available to your key (so you can pick a valid --model);
  2) a raw test chat call (confirms key+model work);
  3) for the first 3 real-heldout rows, each extracted claim's `verification_method` and
     `fallback_used` + the analyzer's verifier-error string.

Interpretation:
  - verification_method == 'llm_claim_entailment_verifier_v1'  -> NLI ENGAGED (good).
  - verification_method in {value_aware..., deterministic_overlap...} or fallback_used=True
    -> it FELL BACK to heuristic (the model errored or wasn't wired) — NLI did NOT run.
This is the missing signal: the native-vs-NLI A/B was identical and we could not tell which.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raggov_score import build_run, _load_rows  # noqa: E402

HELDOUT = (
    Path(__file__).resolve().parent.parent
    / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
)


def _list_models(key: str) -> None:
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        ids = [m.get("id") for m in data.get("data", [])]
        print("Groq models available:", ", ".join(sorted(i for i in ids if i)))
    except Exception as exc:
        print("Could not list models:", type(exc).__name__, str(exc)[:160])


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    args = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set."); return

    print("=== 1. models ==="); _list_models(key)

    from groq_client import GroqClient
    client = GroqClient(model=args.model)
    print("\n=== 2. raw test chat ===")
    try:
        print("model:", args.model, "| reply:", client.chat("Reply with exactly: ok")[:60])
    except Exception as exc:
        print("CHAT FAILED:", type(exc).__name__, str(exc)[:200])
        print("-> pick a valid --model from the list above."); return

    from raggov.engine import DiagnosisEngine
    eng = DiagnosisEngine(config={
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    })
    print("\n=== 3. per-claim engagement (first 3 rows) ===")
    for case in _load_rows(HELDOUT)[:3]:
        d = eng.diagnose(build_run(case))
        for r in getattr(d, "analyzer_results", []) or []:
            if r.analyzer_name != "ClaimGroundingAnalyzer":
                continue
            err = getattr(r, "analyzer_report", None)
            methods = [
                (getattr(c, "label", "?"), getattr(c, "verification_method", None),
                 getattr(c, "fallback_used", None))
                for c in (r.claim_results or [])
            ]
            print(f"  primary={d.primary_failure.value} | claims(label,method,fallback)={methods[:4]}")


if __name__ == "__main__":
    main()
