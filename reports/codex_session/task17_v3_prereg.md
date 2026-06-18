# Task 17-v3 pre-registration — relative vs absolute staleness

**Date:** 2026-06-17 · written BEFORE code. Supersedes attempts 1 & 2 (both
reverted; see task17_result.md and FAILED_APPROACHES.md E/F).

## Root cause (established)

Both the wiki false positives AND the protected case `version_stale_not_cited_32`
reach a STALE `fail` through the same path in `_severity_buckets`:
`STALE_BY_AGE` (absolute age vs an *assumed* "now") + `query_relevant` + retrieved
+ not cited → `retrieved_only_stale` → `retrieval_quality_affected` → fail.

The only honest difference:
- Protected case: the stale doc (`Old CEO Bob (2010)`) coexists with a **newer
  dated version** (`New CEO Alice (2024)`). A genuinely stale retrieval.
- Wiki FPs: a lone old year in prose ("the 1945 film") with **no newer dated
  alternative**. Old-but-correct, not a retrieval failure.

## Change (one, narrow)

In `_severity_buckets`, the `query_relevant` sub-branch of the STALE_BY_AGE
handling additionally requires a **strictly newer dated alternative** among the
other retrieved docs (compared via `issue_date`/`effective_date` on the document
records). If no newer dated version was retrieved, an old-but-relevant doc no
longer escalates to a retrieval failure. All other branches unchanged.

New helper `_has_newer_dated_alternative(records_by_doc_id, doc_id,
retrieved_doc_ids)` — reads only structured record dates, never text or query.

## Measured baseline (BEFORE)

- Protected: 41/46 GREEN (incl. `version_stale_not_cited_32`).
- Calib scored primary: 23/45 = 0.511; gc-012/gc-013 STALE.
- Probe CLEAN: 3/30 correct, 5 STALE false positives.

## Hard acceptance criteria

1. Protected baseline stays **≥ 41/46 GREEN**, including `version_stale_not_cited_32`.
2. Calib scored primary **≥ 0.511**; gc-012 & gc-013 stay STALE_RETRIEVAL.
3. Probe: STALE false positives among the 30 CLEAN **drop materially** (target ≤1),
   and probe CLEAN-correct **strictly increases** (> 3).
4. No new dangerous-miss (no real STALE/failure case flips to CLEAN; safety counters 0).
5. Predicate uses structured record dates only (pipeline/domain-agnostic).

**Revert trigger:** criterion 1, 2, or 4 fails → revert, keep this record.
If criterion 3 cannot reach 0 FPs without breaking 1, document the residual and
the remaining FP cause (don't loosen 1).
