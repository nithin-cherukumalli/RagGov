# Most-important task — Codex (S2): LLM-relabel the heldout from per-claim NLI verdicts

The RAGTruth "contradicted" labels are confirmed mismapped (4 judges agree). With Kimi working, fix
the labels properly. READ-ONLY on engine/policy/canonical dataset; STAGING outputs only; never write
gold or touch the lock.

## Setup
`.env` has `KIMI_API_KEY`; use `--model moonshot-v1-8k`. Reuse `scripts/kimi_client.py` +
`scripts/raggov_score.py:build_run`.

## Granularity rule (Opus's proposed default — flag for the human if you disagree)
Whole-answer primary = most-severe claim-level NLI verdict:
- any claim `contradicted` -> `CONTRADICTED_CLAIM`
- else any `unsupported` -> `UNSUPPORTED_CLAIM`
- else (all verifiable claims `entailed`, rest abstain) -> `CLEAN`
- if the answer extracts no verifiable claims at all -> leave `INSUFFICIENT_CONTEXT` for human review.

## Task
1. Build `scripts/relabel_heldout_from_nli.py` (standalone): for each of the 75 rows, run the engine
   with Kimi `llm_entailment`, collect the per-claim NLI labels, derive the whole-answer label by the
   rule above, and write `evals/govrag_calib/staging/raw/heldout_real_v1_relabeled.jsonl` with:
   original source_label, NLI-derived label, the per-claim label counts, `label_source=llm_assisted_provisional`,
   and `agreement_with_source` (bool). NEVER gold.
2. Quantify the mismapping: of the 25 source=CONTRADICTED rows, how many does NLI re-derive as
   CONTRADICTED vs UNSUPPORTED vs CLEAN? Of the 50 source=CLEAN rows, how many stay CLEAN? This is the
   honest label-quality report.
3. Emit a human spot-audit worklist: all rows where NLI label != source label, plus 10 random agrees,
   for the human to confirm. Do NOT finalize.
4. Append the counts + method to the ledger and `reports/calibration/`.

This gives Opus a heldout with trustworthy labels so the real generalization number (and the
grounded-clean gate's effect) can finally be measured against truth, not noise.

## Closeout (ledger format)
Files inspected; the relabel script + staging file paths; the mismapping counts; the audit worklist;
protected/labels/gates changed (no); next step. HAND BACK — Opus + human accept labels.
