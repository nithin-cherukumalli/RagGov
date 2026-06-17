# Calibration Dataset Quality Audit — dataset_v1

**Date:** 2026-06-17
**Dataset:** `evals/govrag_calib/govrag_calib_150.jsonl` (frozen as `dataset_v1`, SHA256 `fd55e009…`)
**Method:** read-only scan. No labels, cases, or splits were modified.

## Headline numbers

- **52 cases total.** Splits: train 31, dev 11, heldout **3**, unset 7.
- **Label source:** 45 `synthetic_mutation`, 7 `benchmark_migrated`. The dataset is
  overwhelmingly *synthetic*, not real-world pipeline traces — worth stating plainly
  in any accuracy claim.
- **Label confidence:** 42 high, 3 medium, 7 low (the 7 low == the 7 placeholders).

## Finding A — 7 placeholder cases (not real data)

All 7 carry `TODO`/`placeholder` notes, `label_confidence=low`, and `split=unset`
(so they are **not currently scored** — good). They must be filled or dropped, never
silently promoted into a scored split.

| case | type it's standing in for | note |
|---|---|---|
| gc-046 | RETRIEVAL_ANOMALY | "TODO: real near-duplicate retrieval case" |
| gc-047 | RETRIEVAL_DEPTH_LIMIT | "TODO: real depth limit case" |
| gc-048 | LOW_CONFIDENCE | "TODO: real low-confidence borderline case" |
| gc-049 | RETRIEVAL_ANOMALY | "TODO: embedding drift — needs real corpus data" |
| gc-050 | GENERATION_IGNORE | "TODO: need LLM-trace evidence" |
| gc-051 | CITATION_MISMATCH | "TODO: real does_not_support citation case" |
| gc-052 | CITATION_MISMATCH | "TODO: real missing_required citation case" |

(Note: `gc-025` and `gc-044` contain the word "wrong" in their notes but are
legitimate cases describing wrong-tier / phantom-citation scenarios, not placeholders.)

## Finding B — the taxonomy badly outruns the data

Counting only the 45 real (non-placeholder) cases, **11 of 14 active failure types
have fewer than 5 examples**, and 2 have **zero** real cases. A 15th type,
`RERANKER_FAILURE`, is in the enum with **no cases at all** (not even a placeholder).

| failure type | real cases | meets ≥5 floor? |
|---|---|---|
| CONTRADICTED_CLAIM | 11 | ✅ |
| CLEAN | 10 | ✅ |
| INSUFFICIENT_CONTEXT | 5 | ✅ |
| UNSUPPORTED_CLAIM | 4 | ❌ |
| STALE_RETRIEVAL | 3 | ❌ |
| SCOPE_VIOLATION | 3 | ❌ |
| RETRIEVAL_DEPTH_LIMIT | 2 | ❌ |
| CITATION_MISMATCH | 2 | ❌ |
| PROMPT_INJECTION | 2 | ❌ |
| RETRIEVAL_ANOMALY | 1 | ❌ |
| PRIVACY_VIOLATION | 1 | ❌ |
| POST_RATIONALIZED_CITATION | 1 | ❌ |
| LOW_CONFIDENCE | 0 | ❌ |
| GENERATION_IGNORE | 0 | ❌ |
| RERANKER_FAILURE | 0 | ❌ (not in dataset at all) |

**Implication:** only 3 failure types currently have enough real data to make an
accuracy claim mean anything. Everything else is, statistically, anecdote. This is
the single biggest reason the diagnosis cannot yet be called trustworthy.

## Finding C — the heldout split is too small to trust (3 cases)

A 3-case heldout split cannot support a meaningful "heldout accuracy ≥ 0.733"
claim — one case is 33 points. (If headline numbers were computed on a different,
larger heldout file, that file must be identified and locked too.)

## Finding D — unreconciled label drift (see LABEL_CHANGELOG.md)

Labels referenced in `tasks_3_4_5_result.md` (2026-06-15) no longer match the
current goldens for the same case numbers. Recorded in `LABEL_CHANGELOG.md` for
human adjudication; prior accuracy numbers are not comparable to current ones.

## Recommended decisions (need human sign-off — not done here)

1. **Quarantine `RERANKER_FAILURE`** (and optionally `LOW_CONFIDENCE`,
   `GENERATION_IGNORE`) from the public enum / advertised modes until they have
   real cases — they are currently undetectable and unvalidatable.
2. **Set a per-type floor** (suggest ≥5 real cases) for any type that counts toward
   headline accuracy or appears in README claims.
3. **Fill or drop the 7 placeholders** via the changelog process.
4. **Decide the canonical heldout** and grow it to a defensible size.
5. **Adjudicate the drifted labels** in `LABEL_CHANGELOG.md`.

## Guardrails

Read-only audit. The dataset is now frozen + lock-checked (`dataset_v1`). No labels,
cases, or splits were modified by this audit.
