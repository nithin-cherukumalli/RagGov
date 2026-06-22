# Sidekick — local_nli measurement of gate v2 (run on the user's Mac)

**Supersedes the cloud A/B for measurement.** The Groq/Kimi runs hit 70% / 42% rate-limit
fallback at 239 sequential claim calls, so the gate never got a clean entailment-grade signal.
The offline `local_nli` verifier (CrossEncoder) has NO rate limit and NO fallback — use it.

## One-time setup
```
cd <repo>
source .venv/bin/activate         # or your env
export PYTHONPATH=src:.
pip install sentence-transformers  # pulls torch; ~1-2 min. First run downloads the NLI model (~250MB).
```

## Step 1 — clean A/B (expect fallback ~0%)
```
python scripts/run_nli_heldout.py --provider local_nli
# Apple Silicon: add --device mps for speed;  CPU is fine, just slower.
```
Report the two printed lines verbatim: NATIVE and LOCAL-NLI (overall, clean_fp_rate, fallback_pct).
**fallback_pct must be ~0%** — if it's high, sentence-transformers didn't import; fix that first.

## Step 2 — threshold sweep (no code edits)
```
for T in 0.6 0.7 0.75 0.8 0.9; do
  echo "=== threshold $T ==="
  RAGGOV_GROUNDED_CLEAN_ENTAILED_FRACTION=$T python scripts/run_nli_heldout.py --provider local_nli
done
```
For each T report: LOCAL-NLI overall, clean_fp_rate, and the per_type line (so we can see CLEAN
recovery AND whether any CONTRADICTED_CLAIM/gold-FAIL row dropped).

## Acceptance (per prereg `phase2_grounded_clean_gate_v2_prereg.md`)
- PASS: clean_fp_rate drops materially (target ≤ 0.60) at some T, with the CONTRADICTED_CLAIM
  count (gold-FAIL recall) NOT dropping vs the NATIVE line. Pick the lowest CLEAN-FP that holds
  recall flat → Opus sets that as the new default and re-verifies.
- FAIL/REVERT: if no T improves CLEAN-FP without losing gold-FAIL recall → Opus reverts the gate.

## Notes / known limits (so the result is read correctly)
- The gate can only help rows whose native winner is a retrieval-health type (STALE / INCONSISTENT
  / INSUFFICIENT_CONTEXT / SCOPE / RETRIEVAL_ANOMALY / CITATION_MISMATCH) AND that extract ≥1 claim.
  Rows that extract ZERO claims (many ALCE/hotpotqa rows → INSUFFICIENT_CONTEXT in the relabel)
  are NOT reachable by the gate — that's a claim-extraction gap, a separate lever. Don't expect the
  gate alone to fix those.
- Report raw numbers only; do not tune beyond this pre-registered sweep. Opus re-verifies before
  anything is accepted.
```
