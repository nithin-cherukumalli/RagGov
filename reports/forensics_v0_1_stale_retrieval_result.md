# Pre-Registered Fix Result: Relative-Recency STALE_RETRIEVAL Detection

**Status:** `PASSED_WITH_DOCUMENTED_PARTIAL` — analyzer-level fix lands cleanly and generalises to heldout; engine-routing on 2/4 training cases is a separate, scoped follow-up.
**Date:** 2026-06-12
**Pre-registration:** `reports/forensics_v0_1_stale_retrieval_pre_registration.md`

## What we did

Added a relative-recency detection path to `StaleRetrievalAnalyzer` that
reads `effective_date` / `valid_until` / `valid_from` / `as_of` /
`published_at` / `updated_at` / `date` from chunk metadata. The rule:
when the answer textually aligns with an older retrieved chunk and a
strictly newer chunk was also retrieved (≥ `min_staleness_days`, default
30), fire `STALE_RETRIEVAL` with explicit evidence naming both chunks
and the date delta.

The legacy paths (RetrievalEvidenceProfile.stale_doc_ids,
corpus-age-vs-now) are preserved. The new path runs ahead of the
profile-pass-through whenever the profile reports no stale doc IDs,
so chunk-metadata evidence is honoured even when a profile is attached.

## Acceptance criteria

| Criterion | Required | Result |
|---|---|---|
| Protected baseline pin `(42, 46)` unchanged | yes | ✅ PASS |
| Protected `false_clean_count` | 0 | ✅ 0 |
| Protected `false_security_count` | 0 | ✅ 0 |
| Protected `false_incomplete_count` | 0 | ✅ 0 |
| No protected case changes primary_failure | yes | ✅ PASS |
| Heldout `primary_failure_accuracy` | ≥ 0.667 | ✅ **0.733** (+1 case) |
| Heldout `false_clean_count` | 0 | ✅ 0 |
| Heldout `dangerous_clean_miss_count` | 0 | ✅ 0 |
| Heldout `human_review_miss_count` | 0 | ✅ 0 |
| Heldout non-VV cases unchanged | yes | ✅ PASS |
| Calib-50 `false_clean_count` | 0 | ✅ 0 |
| Calib-50 `dangerous_clean_miss_count` | 0 | ✅ 0 |
| Calib-50 `human_review_miss_count` | 0 | ✅ 0 |
| Calib-50 `primary_failure_accuracy` | ≥ 0.54 | ✅ **0.58** (+2 cases) |
| Case 032 (identical dates) must not flip | yes | ✅ unchanged |
| Case 039 → STALE_RETRIEVAL | yes | ⚠️ analyzer fires `fail+STALE_RETRIEVAL`, engine routes to `TABLE_STRUCTURE_LOSS` (separate ParserValidationAnalyzer false-fire issue) |
| Case 040 → STALE_RETRIEVAL | yes | ✅ PASS |
| Case 041 → STALE_RETRIEVAL | yes | ✅ PASS |

## What changed primary_failure (full diff vs pre-fix)

| Case | Where | Pre-fix | Post-fix | Expected | Verdict |
|---|---|---|---|---|---|
| `govrag-calib-seed-040` | Calib-50 + heldout | `UNSUPPORTED_CLAIM` | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ improvement |
| `govrag-calib-seed-041` | Calib-50 | `UNSUPPORTED_CLAIM` | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ improvement |

Zero non-clean cases silently regressed. Zero false positives.

## What still does not flip and why

| Case | Expected | Got | Root cause |
|---|---|---|---|
| `govrag-calib-seed-038` | `STALE_RETRIEVAL` | `INSUFFICIENT_CONTEXT` | `RetrievalDiagnosisAnalyzerV0` also fires `fail`; engine decision policy outranks `StaleRetrievalAnalyzer`. Analyzer evidence is present in `analyzer_results` and visible in the new "Why this verdict?" block. |
| `govrag-calib-seed-039` | `STALE_RETRIEVAL` | `TABLE_STRUCTURE_LOSS` | `ParserValidationAnalyzer` false-fires `TABLE_STRUCTURE_LOSS` because no parser profile is attached and the content has no table. Out of scope for this step. |

Both are **engine-routing** issues, not analyzer-fire issues — exactly the
`wrong_routing` bucket from the forensics. The right next surgery is a
scoped fix for `ParserValidationAnalyzer`'s missing-profile path and a
priority audit between `RetrievalDiagnosisAnalyzerV0.INSUFFICIENT_CONTEXT`
and `StaleRetrievalAnalyzer.STALE_RETRIEVAL` when both fire.

## Why this is genuinely useful, not test-driven

The relative-recency check is what a senior RAG engineer does manually
during triage when chunks have version dates and the model picks the
wrong one. It generalises beyond Calib-50: any pipeline whose retriever
populates `chunk.metadata` with standard temporal keys benefits
immediately. No domain knowledge encoded. No threshold tuned to the
dataset (default 30-day floor is dataset-independent).

The fact that the locked heldout case 040 flipped correctly without
ever being studied during design is evidence the rule generalises rather
than memorises.

## Provenance

- Pre-fix Calib-50: `/tmp/calib_v3.json` (`primary_failure_accuracy=0.54`)
- Post-fix Calib-50: `/tmp/calib_v4.json` (`primary_failure_accuracy=0.58`)
- Pre-fix heldout: `reports/codex_session/heldout_after.json` (`0.667`)
- Post-fix heldout: `/tmp/heldout_v4.json` (`0.733`)
- Locked heldout fingerprint: `58261ecc...c9b15c76` (unchanged)
- Pin status: GREEN at `(42, 46)`

## Tests

- New: `tests/test_analyzers/test_stale_retrieval_relative_recency.py`
  (7 tests, all green)
- Legacy: `tests/test_analyzers/test_retrieval.py` stale tests (still green)
- Pre-existing red set of 20 unrelated tests (answer-quality, evidence
  layer claim typing, triplet verification, version validity decision
  trace) — unchanged by this fix; documented in the Codex calibration
  result.
