# Task 17-v3 result — LANDED

**Date:** 2026-06-17
**Pre-registration:** `reports/codex_session/task17_v3_prereg.md`
**Status:** `LANDED` — all hard revert-trigger criteria met.

## Change

`_severity_buckets` in `TemporalSourceValidityAnalyzerV1`: a query-relevant
`STALE_BY_AGE` document now escalates to a retrieval failure **only when a
strictly newer dated version was also retrieved** (relative recency), via the new
`_has_newer_dated_alternative` helper (structured `effective_date`/`issue_date`
only). Absolute age vs an assumed "now", with no newer dated alternative, no
longer escalates.

## Measured (BEFORE → AFTER)

| Criterion | Required | Before | After | Verdict |
|---|---|---|---|---|
| Protected baseline | ≥ 41/46 GREEN | 41/46 | **41/46** | ✅ |
| `version_stale_not_cited_32` | stays STALE | STALE | **STALE** | ✅ |
| Calib scored primary | ≥ 0.511 | 0.511 | **0.511** | ✅ |
| gc-012 / gc-013 | stay STALE | STALE | **STALE** | ✅ |
| Probe STALE false positives (of 30 CLEAN) | drop materially | 5 | **2** (−60%) | ✅ (residual 2) |
| Probe CLEAN-correct | strictly > 3 | 3 | **4** | ✅ |
| Dangerous-miss / safety counters | 0 | 0 | **0** | ✅ |
| Analyzer unit tests | green | green | **166 passed, 1 xfailed** | ✅ |

## Residual (documented, per pre-registration)

2 of the original 5 STALE false positives remain. These are wiki contexts where
another retrieved chunk genuinely carries a newer parsed date than the stale
chunk, so the relative-recency signal fires. This is a more defensible call than
the lone-old-year cases that were removed, and eliminating it fully would require
entity/topic grouping (knowing the two dated docs are about the *same* thing),
which is a larger change. Filed as a note, not loosened.

The broader CLEAN over-firing is only partly closed by this task: with STALE
suppressed, some CLEAN cases now mis-route to `INCONSISTENT_CHUNKS` (8) and
`CHUNKING_BOUNDARY_ERROR` (5) — **separate** over-firing analyzers, out of scope
here. Those are the next precision targets (follow-up).

## Why this one landed when 1 & 2 didn't

Attempts 1 (skip analyzer) and 2 (delete text inference) were too blunt — they
disabled load-bearing behaviour and broke a pinned case. 17-v3 keeps the analyzer
and its text parsing intact and changes only the **escalation rule**, separating
the legitimate relative-recency signal (which the pinned case needs) from the
absolute-age false positive (which it doesn't). First engine precision fix to
earn its landing under the pre-registered criteria.
