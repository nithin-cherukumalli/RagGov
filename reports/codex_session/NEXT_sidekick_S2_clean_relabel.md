# Sidekick S2 (Codex) — clean NLI relabel re-run (run on the user's Mac)

**Role:** read-only measurement/data. You PREPARE numbers and provisional data; Opus verifies and
decides. Do NOT edit engine/policy code, do NOT write gold/canonical datasets, do NOT accept your
own output as truth.

**Why now:** the prior relabel (`heldout_real_v1_nli_relabel_report.md`) was 101/239 claim checks
on fallback (~42%), so it's provisional and untrustworthy. Commit `ee0b433` added backoff retry to
the clients. Re-run the relabel cleanly to see if fallback dropped enough to make the relabel
adjudication-ready. This is independent of S1's gate sweep — run it in parallel.

## Setup
```
cd <repo>
export PYTHONPATH=src:.
# .env holds groq_api / KIMI_API_KEY (rotate first if not already done).
```

## Step 1 — re-run the provisional relabel (now with backoff)
```
python scripts/relabel_heldout_from_nli.py --model moonshot-v1-8k
```
This writes (provisional, gitignored staging):
- `evals/govrag_calib/staging/raw/heldout_real_v1_relabeled.jsonl`
- `evals/govrag_calib/staging/raw/heldout_real_v1_nli_spot_audit_worklist.jsonl`
- `reports/calibration/heldout_real_v1_nli_relabel_report.md`

## Step 2 — report these numbers verbatim (do not interpret/accept)
1. **Fallback rate:** how many of the N claim checks used `fallback_used` / hit an invoke error,
   vs the prior 101/239 (~42%). This is the headline — did backoff fix it?
2. **Relabel confusion vs source labels:** of the 25 source-CONTRADICTED rows, how many re-derive
   CONTRADICTED / UNSUPPORTED / INSUFFICIENT / CLEAN; of the 50 source-CLEAN, how many stay CLEAN
   vs move. (Prior run: 25→ {10C, 8U, 6I, 1CLEAN}; 50→ {22 CLEAN, 20 INSUFF, 5 UNSUP, 3 CONTRA}.)
3. **Worklist size:** how many rows landed on the spot-audit worklist for human review.

## Decision gate (Opus owns it)
- If fallback is now low (say < 10%): the relabel is adjudication-ready → Opus + human work the
  worklist to lock a trustworthy heldout (handoff Step C).
- If fallback is still high: backoff alone wasn't enough → flag it; the next lever is the cheap
  half of Step B (risk-filtered / batched verification), which Opus builds before we trust any
  relabel or gate number.

## Hard rules
- Report raw counts only; no tuning, no relabeling of CONTRADICTED rows toward the source labels
  (they're known-mismapped — 4 judges agree). The relabel is to DISCOVER truth, not match the old.
- Opus re-verifies every number before anything is accepted.
```
