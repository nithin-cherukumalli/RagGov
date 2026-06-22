# Codex task — adjudicate the heldout into a LOCKED gold set (do this carefully)

You are the careful second judge. The provisional benchmark labels are broken (the 25
"CONTRADICTED_CLAIM" rows are mislabeled — a first Kimi pass found 0/25 actually contradicted).
Your job: read each of the 75 heldout rows yourself and assign the correct v1 label, producing a
trustworthy gold set. Opus + Nithin will verify your output before it is accepted as final.

## Context you need
- RagGov is a RAG **failure-attribution / governance auditor**: given (query, retrieved chunks,
  answer), it says which stage failed and why. We judge faithfulness **only against the provided
  chunks** — never outside knowledge.
- Files (repo root `/Users/nitin/Desktop/RagGov`, `source .venv/bin/activate`, `export PYTHONPATH=src:.`):
  - Source rows: `evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`
    (fields: `case_id`, `query`, `retrieved_chunks` [{chunk_id, doc_id, text}], `answer`,
    `citations`, `expected_primary_failure` = the BROKEN provisional label).
  - Kimi's first-pass proposal: `..._gold_kimi.jsonl` (use as a hint, NOT as truth).
  - Editable sheet: `..._gold_sheet.csv` (you will fill the `final_label` column).
  - Taxonomy reference: `reports/codex_session/taxonomy_v1.md`.

## The 6 v1 labels (the ONLY valid `final_label` values)
- `CLEAN` — answer is faithful and adequately supported by the chunks (paraphrase is fine).
- `PROMPT_INJECTION` — query/chunk tries to hijack the model, or the answer obeys injected
  instructions. Security-first.
- `STALE_RETRIEVAL` — chunks are an outdated/superseded version; the answer reflects stale info.
- `INSUFFICIENT_CONTEXT` — chunks don't contain enough to answer; answer is incomplete or guesses.
- `UNSUPPORTED_CLAIM` — answer asserts specific facts that appear in NONE of the chunks
  (fabrication), without directly contradicting them.
- `CONTRADICTED_CLAIM` — answer DIRECTLY conflicts with what a chunk says (e.g. chunk says "2019",
  answer says "2021"). "Not in the chunks" is UNSUPPORTED, NOT contradicted — keep this distinction
  strict; it is the mistake the original benchmark made.

Precedence when several apply: PROMPT_INJECTION > CONTRADICTED_CLAIM > UNSUPPORTED_CLAIM >
STALE_RETRIEVAL > INSUFFICIENT_CONTEXT > CLEAN. Prefer CLEAN when the answer is reasonably supported.

## Two judgment rules to apply consistently
1. **ALCE/QAMPARI list-answers** (`case_id` contains `alce`): these list many entities; the heldout
   only carries top-k chunks, so some listed entities aren't in the provided evidence. Per the
   faithfulness definition, an entity not in any provided chunk is **UNSUPPORTED_CLAIM**. Default to
   UNSUPPORTED for these unless ALL listed items are clearly present in the chunks (then CLEAN).
   Note this is a definitional call — flag it clearly so Nithin can confirm the policy.
2. **ex-CONTRADICTED rows** (`provisional == CONTRADICTED_CLAIM`): re-judge from scratch. Almost all
   are CLEAN, UNSUPPORTED, or INSUFFICIENT — only mark CONTRADICTED if you find a DIRECT conflict
   with a chunk and can quote both sides.

## Procedure
1. Make the sheet (if not present):
   `python scripts/gold_adjudication.py sheet --from evals/govrag_calib/staging/raw/heldout_real_v1_gold_kimi.jsonl`
2. For EACH of the 75 rows: open the corresponding source row, read the query + every chunk + the
   answer, and set `final_label` in the CSV to the correct v1 label. Do NOT blindly copy Kimi —
   verify against the chunks. Keep a one-line reason per row where you change Kimi's label.
3. Finalize: `python scripts/gold_adjudication.py finalize`
   (builds the locked gold + prints distribution, exact accuracy, CLEAN-FP, detection rate).

## Deliverables (report these back)
- The `finalize` output (gold distribution + the 4 numbers).
- A list of every row where you DISAGREED with Kimi, with your label + one-line reason.
- A short "uncertain — Nithin please confirm" list (anything genuinely ambiguous + the ALCE policy).
- Confirm: zero rows left as the old CONTRADICTED unless you can quote the direct conflict.

## Hard rules (discipline — this is the product's credibility)
- Judge ONLY against the provided chunks. No outside knowledge.
- Do NOT modify the engine, analyzers, or any canonical dataset; only the sheet + locked gold.
- Do NOT tune anything to the numbers — this is the ruler, not a training set.
- When unsure, mark it uncertain and flag it; never guess silently.
- Every `final_label` must be one of the 6 valid labels or `finalize` will reject the sheet.
