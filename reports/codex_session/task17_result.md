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
