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
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raggov_score import build_run, _load_rows  # noqa: E402
from groq_client import DEFAULT_GROQ_MODEL, _load_key  # noqa: E402
from raggov.connectors.groq_client import GroqLLMClient  # noqa: E402

HELDOUT = (
    Path(__file__).resolve().parent.parent
    / "evals" / "govrag_calib" / "staging" / "raw" / "heldout_real_v1.jsonl"
)


def _redact(text: str, key: str | None) -> str:
    if not key:
        return text
    return text.replace(key, "[REDACTED]")


def _http_error_details(exc: Exception, key: str | None = None) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        detail = f"HTTP {exc.code} {exc.reason}"
        if body:
            detail += f" | body={body[:500]}"
        return _redact(detail, key)
    return _redact(f"{type(exc).__name__}: {str(exc)[:500]}", key)


def _print_http_guidance(exc: Exception) -> None:
    lowered = str(exc).lower()
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {401, 403}:
            print(
                "-> auth/authorization failure, not a Hugging Face issue. "
                "Create/rotate a Groq API key, export it as GROQ_API_KEY, and ensure "
                "the key is enabled for API use in your Groq project."
            )
            return
        if exc.code == 400:
            print(
                "-> request/model error. Try --model llama-3.1-8b-instant or another model "
                "available to your Groq account."
            )
            return
        if exc.code == 429:
            print("-> rate limited. Wait or use a project/key with available quota.")
            return
    if "error code: 1010" in lowered:
        print(
            "-> Groq/Cloudflare blocked this client/IP signature (1010). "
            "The script now uses the official Groq SDK; if this persists, rotate the key "
            "and try from a normal home/mobile network without VPN/proxy."
        )
        return
    if "403" in lowered or "401" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        print(
            "-> auth/authorization failure, not a Hugging Face issue. "
            "Rotate/create a Groq API key and export it as GROQ_API_KEY with no spaces."
        )
        return
    if "429" in lowered or "rate limit" in lowered:
        print("-> rate limited. Wait or use a project/key with available quota.")
        return
    print("-> see the error detail above; model listing is optional, but chat must pass.")


def _list_models(key: str) -> None:
    try:
        from groq import Groq

        client = Groq(api_key=key)
        data = client.models.list()
        ids = [getattr(model, "id", None) for model in getattr(data, "data", [])]
        print("Groq models available:", ", ".join(sorted(i for i in ids if i)))
        return
    except ImportError:
        print("Groq SDK not installed; falling back to raw HTTP model listing.")
    except Exception as exc:
        print("Could not list models via Groq SDK:", _http_error_details(exc, key))
        _print_http_guidance(exc)
        return

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
        print("Could not list models:", _http_error_details(exc, key))
        _print_http_guidance(exc)


def main() -> None:
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    ap.add_argument(
        "--skip-model-list",
        action="store_true",
        help="skip /models; useful if your key can chat but cannot list models",
    )
    args = ap.parse_args()
    key = _load_key()
    if not key:
        print(
            "GROQ_API_KEY not set. Export it or add `groq_api = gsk_...` to a gitignored .env file."
        )
        return

    print("=== 1. key/env ===")
    print(f"key source: found ({len(key)} chars, redacted)")
    print(f"model: {args.model}")

    print("\n=== 2. models ===")
    if args.skip_model_list:
        print("Skipped model listing.")
    else:
        _list_models(key)
    print("\n=== 3. raw test chat ===")
    try:
        raw_client = GroqLLMClient(
            api_key=key,
            model=args.model,
            temperature=0.0,
            max_tokens=32,
            json_mode=False,
        )
        print("model:", args.model, "| reply:", raw_client.chat("Reply with exactly: ok")[:60])
    except Exception as exc:
        print("CHAT FAILED:", _http_error_details(exc, key))
        _print_http_guidance(exc)
        if "missing_groq_package" in str(exc).lower() or "no module named" in str(exc).lower():
            print("-> install the SDK in this venv: pip install groq")
        return

    from raggov.engine import DiagnosisEngine
    client = GroqLLMClient(
        api_key=key,
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
        json_mode=True,
    )
    eng = DiagnosisEngine(config={
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    })
    print("\n=== 4. per-claim engagement (first 3 rows) ===")
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
