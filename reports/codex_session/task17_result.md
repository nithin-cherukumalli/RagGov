# Task 17 result — attempt 1 FAILED, REVERTED

**Date:** 2026-06-17
**Pre-registration:** `reports/codex_session/task17_prereg.md`
**Status:** `REVERTED` — the change failed its hard acceptance criteria.

## What was tried

Added a precondition to `TemporalSourceValidityAnalyzerV1.analyze`: if no
structured temporal/version signal exists in the evidence, `skip` the analyzer
(which also starves `RetrievalDiagnosisAnalyzerV0`, since it consumes
`run.version_validity_report`).

## What was measured

| Criterion | Required | After change | Verdict |
|---|---|---|---|
| Protected baseline | ≥ 41/46 GREEN | **version_validity 0/5 → FAIL** | ❌ |
| Calib scored primary | ≥ 0.511 | **0.356** | ❌ |
| Probe STALE false positives (CLEAN) | 5 → 0 | 5 → 0 | ✅ |
| Probe CLEAN-correct | > 3 | **0** (worse) | ❌ |
| gc-012 / gc-013 stay STALE | yes | yes | ✅ |

The STALE false positives did vanish — but **skipping the whole analyzer was too
blunt**. Two regressions resulted:
1. The protected baseline expects `version_validity` to run on 5/5 of its cases;
   skipping broke that (0/5).
2. With no version-validity report, the engine fell through to
   `INCOMPLETE_DIAGNOSIS` (5×) and more `INCONSISTENT_CHUNKS` on the CLEAN probe
   cases — so CLEAN-correct went 3 → 0. We traded one false-positive class for
   another.

Per the pre-registered revert trigger (criteria 1, 2, 4), the change was reverted.
The pre-registration is preserved as the historical record. This is the same
posture as the v1 Tasks 3/4/5 revert: failed criteria → revert, no silent
loosening.

## Why it failed (root cause of the failed fix)

The analyzer is load-bearing: the engine and the protected baseline depend on it
running and producing a report. The real defect is narrower than "the analyzer
fires without a signal" — it is that **`_text_lifecycle_metadata` turns plain
passage text into lifecycle metadata** (notably `\((\d{4})\)` → `issue_date`),
and that text-only inference escalates to a STALE `fail`. The analyzer must keep
running; only the *text-only-derived* staleness should be prevented from
escalating.

## Reformulation → Task 17-v2 (queued, not yet attempted)

**Hypothesis.** Keep `TemporalSourceValidityAnalyzerV1` running and producing its
report. Change only the escalation rule: a document whose stale/invalid status
derives **solely** from text-parsed lifecycle hints (no structured metadata
timestamp/version for that document) must **not** escalate to a STALE `fail` —
at most a non-promoting `warn`. Equivalent option: drop the most aggressive
text inference (`(YYYY)` → `issue_date`) and re-measure.

**Same hard criteria as task17_prereg.md** (protected ≥ 41/46; Calib ≥ 0.511;
gc-012/013 stay STALE; probe STALE FP → 0 **and** probe CLEAN-correct strictly
increases — this time the second half of criterion 3 must hold, which attempt 1
violated).

**Out of scope:** unchanged.

---

## Attempt 2 (Task 17-v2) — FAILED, REVERTED

**Change.** Kept the analyzer running; in `_text_lifecycle_metadata`, replaced the
aggressive `(YYYY)` / "as of YYYY" → `issue_date` inference with an explicit
"issued/published <year>" inference only.

**Measured.**

| Criterion | Required | After | Verdict |
|---|---|---|---|
| Protected baseline | ≥ 41/46 | **40/46 (case `version_stale_not_cited_32` regressed)** | ❌ |
| Calib scored primary | ≥ 0.511 | 0.511 | ✅ |
| Probe STALE FP (CLEAN) | → 0 | 5 → 1 | partial |
| Probe CLEAN-correct | > 3 | 3 → 5 | ✅ |
| gc-012 / gc-013 | stay STALE | yes | ✅ |

Closer (probe genuinely improved, Calib held) but **criterion 1 failed** →
reverted.

## The real root cause (why a quick fix can't work)

The protected case `version_stale_not_cited_32` is:

> query "Who is the CEO?", chunks `"Old CEO Bob (2010)"` + `"New CEO Alice
> (2024)"`, answer cites Alice. Expected: STALE_RETRIEVAL (a stale doc was
> retrieved even though the answer avoided it).

It detects staleness **from the same parenthetical-year text inference** that
produces the wiki false positives. So the over-firing and a pinned expectation
are entangled in one heuristic.

The distinction that actually separates the good case from the false positives:
- **Relative recency** — two competing docs about the same entity with different
  years (2010 vs 2024) → the older is stale. *Legitimate* (the protected case).
- **Absolute age vs an assumed "now"** — a single year mention in ordinary prose
  ("the 1945 film", "(1889)") aged against today → STALE_BY_AGE. *The false
  positive.*

The current analyzer collapses both into the same text→`issue_date`→age path.

## Status: Task 17 needs a redesign (17-v3), not a patch

**17-v3 hypothesis (queued, needs own pre-registration).** Split the two paths:
keep text-year staleness ONLY when a newer competing version of the same
doc/entity is present (relative recency); do **not** escalate absolute
age-vs-assumed-now staleness to a STALE `fail`. This should preserve
`version_stale_not_cited_32` (relative) while removing the wiki false positives
(absolute). Same hard criteria as above. This is analyzer logic surgery in
`_severity_buckets` / `_document_record`, not a one-line change, so it gets a
fresh pre-registration before any code.

Two disciplined reverts here are the system working as intended: the pinned
criteria caught both regressions before anything false landed.
