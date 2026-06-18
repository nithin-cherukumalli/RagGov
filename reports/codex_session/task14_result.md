# Task 14 result — stale irrelevant source should not primary-fail — LANDED

**Date:** 2026-06-18. Prereg: `task14_prereg.md`.

## What changed
`src/raggov/analyzers/retrieval/stale.py` (`_from_profile`): an age-stale doc now counts only when
BOTH (1) it is query-relevant (≥1 chunk labeled RELEVANT/PARTIAL in the evidence profile) AND
(2) a strictly-newer dated retrieved alternative exists (`run.corpus_entries` timestamps). Missing
relevance/date info → not suppressed (legacy preserved). Added `_stale_doc_is_diagnostic`,
`_doc_query_relevant`, `_has_newer_dated_alternative`. NCV's version_validity node consumes the
StaleRetrievalAnalyzer result, so suppression cascades correctly.

## Acceptance criteria
| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Protected baseline | **43/46 effective, check PASS** | PASS |
| 2 | Calib scored primary | **23/45 (0.511)** unchanged | PASS |
| 3 | Probe overall + no STALE TP flip | **80/145** unchanged | PASS |
| 4 | version_validity TPs (180/190/220/250) + suites green | **12/12 version-validity; 750 passed, 2 xfailed**, 1 pre-existing stale fail | PASS |

**Result:** `test_stale_irrelevant_source_does_not_primary_fail` (case 299) was strict-xfail; the fix
makes it pass, so the xfail marker was removed (now a normal passing test). Genuine stale cases —
relevant outdated version with a newer alternative (250), expired/withdrawn cited sources (190/220),
invalid source (180) — all stay STALE.

## Separator (why this is precise, not a hack)
- Keep (250): stale `doc-old` is RELEVANT (old CEO version) + newer alt `doc-new` → genuine stale.
- Suppress (299): stale `doc-lease` is IRRELEVANT noise; the fresh cited doc has no newer alternative.
The gate reads structured relevance + record dates only (domain-agnostic, pipeline-level — in-bounds
for discipline rule #3). `TemporalSourceValidityAnalyzerV1` already handled 299; the over-firer was
the profile path, now fixed.

## Residual (honest)
Probe CLEAN→STALE_RETRIEVAL false positives remain 3 (unchanged) — those flow through a different
path (no corpus dates → newer-alt gate intentionally does not suppress). Separate from this fix;
left as a documented residual.
