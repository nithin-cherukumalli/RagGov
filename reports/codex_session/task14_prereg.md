# Task 14 pre-registration — stale irrelevant source should not primary-fail

**Date:** 2026-06-18 · written BEFORE code.

## Root cause (instrumented)
`StaleRetrievalAnalyzer._from_profile` fails STALE on **every** `profile.stale_doc_ids` entry, which
is computed from ABSOLUTE age. NCV's version_validity node consumes that result and cascades.

Separator (confirmed on the two pipeline tests):
- Keep (test 250, `test_stale_retrieved_only_blocks_clean_when_retrieval_quality_affected`):
  stale `doc-old` is query-RELEVANT (old CEO version), not noisy, and a strictly-newer dated
  alternative (`doc-new`) was retrieved → genuine stale.
- Suppress (test 299, Task 14 xfail): stale `doc-lease` is query-IRRELEVANT ("office lease", in
  `noisy_chunk_ids`); the only relevant stale doc (`doc-new`) has NO newer alternative (it is the
  newest, status active).

`TemporalSourceValidityAnalyzerV1` already suppresses 299 correctly (it reports `stale_irrelevant`);
the over-firer is the profile path in `StaleRetrievalAnalyzer`.

## Change (one, narrow)
In `StaleRetrievalAnalyzer._from_profile`, count a stale doc only if BOTH:
1. query-relevant — at least one of its chunks has `query_relevance_label` RELEVANT/PARTIAL (i.e.
   not an all-noisy/irrelevant distractor), and
2. a strictly-newer dated retrieved alternative exists (via `run.corpus_entries` timestamps).
Missing relevance/date info → do not suppress (preserve legacy behavior). If no stale doc survives
→ pass. Reads structured profile + record dates only (domain-agnostic, pipeline-level).

## Hard acceptance criteria
1. Protected baseline stays **43/46** (check pass). *(revert)*
2. Calib scored primary stays **≥ 23/45**; gc STALE golds stay STALE. *(revert)*
3. Probe overall stays **≥ 80/145**; no STALE true-positive flips to non-STALE on probe. *(revert)*
4. `tests/test_analyzers/test_version_validity_pipeline.py` genuine-stale TPs (180, 190, 220, 250)
   stay green; full `test_analyzers` green (modulo pre-existing stale fail). *(revert)*

**Success:** `test_stale_irrelevant_source_does_not_primary_fail` (299) flips xfail→pass.
