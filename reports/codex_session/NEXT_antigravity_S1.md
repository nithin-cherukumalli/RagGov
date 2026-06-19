# Most-important task — Antigravity (S1): run the REAL Kimi NLI A/B on the heldout

You are on a networked machine (the LLM works here; Opus's sandbox proxy-blocks it). This is the
measurement Opus cannot run. READ-ONLY on engine/policy/labels/gates.

## Setup
`.env` already has `KIMI_API_KEY`. Kimi works with `--model moonshot-v1-8k`.

## Task — produce the first real native-vs-NLI numbers
1. Run the full A/B on the locked real heldout (75 rows):
   ```bash
   PYTHONPATH=src:. python scripts/run_nli_heldout.py --provider kimi --model moonshot-v1-8k
   ```
   It prints NATIVE vs LLM-ENTAILMENT: overall, per-type, and CLEAN-FP rate.
2. Report a clean before/after table: overall, CLEAN (and CLEAN-FP rate), CONTRADICTED, UNSUPPORTED,
   INSUFFICIENT. Note runtime + any rate-limit retries (75 rows × extractor + per-claim calls is many
   requests — if it's too slow, run on the first 30 rows and say so).
3. **Interpret honestly, expecting two things:** (a) CLEAN-FP likely stays ≈0.76 because swapping the
   grounding verifier doesn't touch the stale/sufficiency/inconsistency analyzers that cause it
   (Opus proved this) — that is the expected result, not a bug; (b) grounding-path accuracy
   (UNSUPPORTED) may shift. The number that matters for Opus's next change is whether the NLI claim
   labels are stable/usable as the "grounded-clean" signal.
4. Also dump, for 10 CLEAN rows, the per-claim NLI labels (entailed/unsupported/contradicted/abstain)
   so Opus can calibrate the grounded-clean gate threshold (how many CLEAN answers have zero
   contradicted claims and mostly-entailed verifiable claims).
5. Write results to `reports/calibration/eval_report_<date>.md` and the ledger. Spot-parity 1 case
   vs `raggov_score`.

## Closeout (ledger format)
Files inspected; the A/B tables; the 10-row per-claim CLEAN dump; runtime/limits;
protected/labels/gates changed (no); next step. HAND BACK — Opus re-verifies before acting.
