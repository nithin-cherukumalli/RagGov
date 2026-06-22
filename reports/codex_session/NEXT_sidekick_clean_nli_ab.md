# Sidekick task — clean NLI A/B + threshold sweep (run on the user's Mac)

**Why:** the build sandbox proxy-blocks Groq/Kimi/HF, so this measurement must run off-sandbox.
Two engine changes just landed that need a CLEAN (low-fallback) NLI run to measure:
- `ee0b433` — bounded backoff retry on 429/5xx in both LLM clients (should cut the ~42% fallback).
- `72fa109` — grounded-clean gate v2: entailed-fraction rule (default threshold 0.75), env-tunable.

## Setup
```
cd <repo>
export PYTHONPATH=src:.
# .env already holds groq_api / KIMI_API_KEY (rotate first if not yet done).
```

## Step 1 — confirm the backoff cut the fallback
Run the A/B once and capture, from the run logs, the fraction of claim checks that hit
`fallback_used` / `llm_entailment_invoke_failed`. Compare to the prior ~42%.
```
python scripts/run_nli_heldout.py --provider kimi      # or --provider groq
```
Record: native line, LLM-ENTAILMENT line (overall + clean_fp_rate), and fallback %.

## Step 2 — threshold sweep (no code edits needed)
The gate threshold is env-overridable. Sweep and record CLEAN-FP **and** gold-FAIL recall at each:
```
for T in 0.6 0.7 0.75 0.8 0.9; do
  echo "=== threshold $T ==="
  RAGGOV_GROUNDED_CLEAN_ENTAILED_FRACTION=$T python scripts/run_nli_heldout.py --provider kimi
done
```

## Acceptance (per prereg `phase2_grounded_clean_gate_v2_prereg.md`)
- PASS: CLEAN-FP drops materially (target ≤ 0.60) at some T, with **zero** of the 25 gold-FAIL
  rows flipping to CLEAN (gold-FAIL recall must not drop). Pick the lowest CLEAN-FP that holds
  recall flat — that T becomes the new default (Opus updates the constant + re-verifies).
- FAIL/REVERT: if no T improves CLEAN-FP without losing gold-FAIL recall, the loosening doesn't
  earn its place — report it and Opus reverts `72fa109`.

## Hard rules
- Do NOT tune on the heldout beyond this pre-registered sweep; report raw numbers.
- Opus re-verifies every number before it's accepted (sidekick numbers are inputs, not verdicts).
- If fallback is still high after Step 1, the gate signal is still degraded — flag it; the
  next lever is the *cheap* half of Step B (risk-filtered / batched verification) before trusting
  the sweep.
