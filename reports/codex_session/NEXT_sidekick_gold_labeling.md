# Sidekick / user task — build the locked gold heldout (two-judge protocol)

Runs on the Mac (Kimi + Groq keys in `.env`; the build sandbox proxy-blocks both). Goal: a
trustworthy, defensible gold heldout to replace the mislabeled provisional one. Protocol = two
independent LLM judges + human adjudication of disagreements only.

## Why two judges
One model's labels aren't credible for a flagship benchmark. Two independent models that AGREE give
a strong provisional gold; only their DISAGREEMENTS need your eyes. This is a standard, defensible
labeling protocol — and it's a great line in the README.

## Step 1 — Labeler A (Kimi), re-run to clear the 7 errors
The script now has a JSON repair-retry, so re-running shrinks the 7 errored rows.
```
PYTHONPATH=src:. python scripts/label_heldout_gold.py --provider kimi \
  --output evals/govrag_calib/staging/raw/heldout_real_v1_gold_kimi.jsonl
```

## Step 2 — Labeler B (Groq, independent provider + bigger model)
```
PYTHONPATH=src:. python scripts/label_heldout_gold.py --provider groq \
  --output evals/govrag_calib/staging/raw/heldout_real_v1_gold_groq.jsonl
# default model llama-3.3-70b-versatile, throttle auto-raised to 1s for Groq RPM limits
```

## Step 3 — Merge
```
PYTHONPATH=src:. python scripts/merge_gold_labels.py \
  --a evals/govrag_calib/staging/raw/heldout_real_v1_gold_kimi.jsonl \
  --b evals/govrag_calib/staging/raw/heldout_real_v1_gold_groq.jsonl
```
Read `reports/calibration/heldout_real_v1_gold_merge_report.md`:
- **raw inter-annotator agreement** = headline trust metric (report it; aim to understand, not hit a target).
- `heldout_real_v1_gold_agreed.jsonl` = both judges agree + confident → provisional gold.
- `heldout_real_v1_gold_review.jsonl` = the ONLY rows you hand-adjudicate.

## Step 4 — Human adjudication (one evening, the review set only)
For each review row, read the query + chunks + answer and pick the v1 label (taxonomy_v1.md).
**Two judgment calls you must make as the product owner (not the model):**
1. **ALCE/QAMPARI list-answers** (rows ~56–75): the answer lists entities; some aren't in the
   included top-k chunks. Kimi called these UNSUPPORTED. Decision: for a *faithfulness/governance*
   auditor, "claim not in the provided evidence" IS a failure → UNSUPPORTED is correct, and the
   old CLEAN labels were too generous. Recommend accepting UNSUPPORTED here, but eyeball 3–4.
2. **The 25 ex-CONTRADICTED rows**: zero held up as contradicted. Confirm a handful by hand so you
   trust the wipeout, then accept the re-derived labels (CLEAN / UNSUPPORTED / INSUFFICIENT).

## Step 5 — Lock the gold + re-score
Merge your adjudicated review labels with the agreed set → `heldout_real_v1_gold.jsonl` (LOCKED).
Then re-score the engine against it. Send me (Opus) the merge report + the locked label distribution
and I'll verify, then we move to Phase B (wire the 6-bucket taxonomy + re-measure honestly).

## Hard rules
- Don't tune the engine on this; it's the ruler, not a training set.
- Report raw agreement + your adjudication decisions; Opus re-verifies before anything is "gold".
- The locked gold replaces `heldout_real_v1.jsonl` as the eval target but does NOT delete it (keep
  provisional for provenance).
```
