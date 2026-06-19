"""Diagnose whether the Kimi (Moonshot) entailment tier ENGAGES on the real heldout.

Run on a machine with network (sandbox proxy-blocks external APIs):

    KIMI_API_KEY=... PYTHONPATH=src:. python scripts/kimi_nli_diagnose.py
    KIMI_API_KEY=... PYTHONPATH=src:. python scripts/kimi_nli_diagnose.py --model moonshot-v1-8k

Prints: available models, a raw test chat, and per-claim verification_method/fallback for
the first 3 real-heldout rows. If verification_method == 'llm_claim_entailment_verifier_v1'
the NLI tier engaged; a heuristic method or fallback_used=True means it fell back.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raggov_score import build_run, _load_rows  # noqa: E402
from kimi_client import KimiClient, DEFAULT_KIMI_MODEL  # noqa: E402

HELDOUT = (
    Path(__file__).resolve().parent.parent
    / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
)


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_KIMI_MODEL)
    ap.add_argument("--skip-model-list", action="store_true")
    args = ap.parse_args()

    try:
        client = KimiClient(model=args.model)
    except Exception as exc:
        print("KIMI client init failed:", str(exc)[:200]); return

    print("=== 1. models ===")
    if args.skip_model_list:
        print("skipped")
    else:
        try:
            print("Kimi models:", ", ".join(client.list_models()))
        except Exception as exc:
            print("could not list models:", str(exc)[:160])

    print("\n=== 2. raw test chat ===")
    try:
        print("model:", args.model, "| reply:", client.chat("Reply with exactly: ok")[:60])
    except Exception as exc:
        print("CHAT FAILED:", str(exc)[:200])
        print("-> check key/model; try --model moonshot-v1-8k"); return

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
            methods = [
                (getattr(c, "label", "?"), getattr(c, "verification_method", None),
                 getattr(c, "fallback_used", None))
                for c in (r.claim_results or [])
            ]
            print(f"  primary={d.primary_failure.value} | claims={methods[:4]}")


if __name__ == "__main__":
    main()
