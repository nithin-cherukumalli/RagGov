# Task 20 result — CHUNKING_BOUNDARY_ERROR over-firing — LANDED

**Date:** 2026-06-18. Prereg: `task20_prereg.md`.

## What changed
`src/raggov/analyzers/parsing/parser_validation.py`: `_has_chunk_boundary_damage` now
receives chunks (not bare texts) and requires the consecutive pair to share the same
`source_doc_id` before flagging a split-sentence boundary error. Multi-hop retrieval (every
chunk a distinct document) no longer triggers it; genuine within-document adjacent-fragment
splits still do. Dangling-word + lowercase-start logic otherwise unchanged.

## Acceptance criteria
| # | Criterion | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Protected baseline | 41/46 | **41/46** | PASS |
| 2 | Calib scored primary | 23/45 (0.511) | **23/45 (0.511)** | PASS |
| 3 | Parsing/engine/policy suites | green* | **green*** | PASS |
| 4 | Probe overall accuracy | 47/145 (0.324) | **54/145 (0.372)** | PASS (strict ↑) |
| 5 | CLEAN→CHUNKING_BOUNDARY_ERROR FPs | 5 | **0** (target ≤1) | MET |

\* 3 pre-existing engine-CLI test failures (clean_pass / prompt_injection / malformed_json)
are red at baseline before this change — unrelated CLI-output drift, out of scope.

CLEAN-correct rose 8/30 → **10/30**. Probe overall +7 (the freed boundary cases cascaded:
+2 to correct CLEAN, the rest to their correct non-CLEAN types — INSUFFICIENT_CONTEXT etc.).

## Tests
- `test_chunk_boundary_damage_flagged_within_same_document` (TP: same `source_doc_id`).
- `test_chunk_boundary_damage_ignored_across_distinct_documents` (precision: distinct docs).

## Note
Expected cascade (per plan): one freed CLEAN case shifted to INCONSISTENT_CHUNKS (now 5 on
CLEAN) and one to STALE — these belong to the negation-residual / Jaccard (Task 21) and STALE
(Task 14/17 residual) mechanisms, not this fix.
