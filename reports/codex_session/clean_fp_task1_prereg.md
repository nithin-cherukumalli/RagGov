# CLEAN-FP increment 1 — demote InconsistentChunksAnalyzer (pre-registration)

**Date:** 2026-06-22 · written BEFORE code.

## Evidence (why this is safe)
`InconsistentChunksAnalyzer` produces **zero true-positives on every dataset we can measure**:
- protected-46 triage: 0 INCONSISTENT_CHUNKS expectations
- Calib (150): 0 gold INCONSISTENT_CHUNKS
- real heldout (75): 0 gold, 0 TP — but **8 CLEAN false-positives**
- synthetic probe (145): 0 gold; the 3 rows where it wins primary are gold CLEAN(2) / INSUFFICIENT(1) — all FP

It is an uncalibrated heuristic contributing only noise. Task 19 already tightened it (8→2 probe FP)
but it still over-fires on real data with no offsetting recall anywhere.

## Hypothesis
Making `InconsistentChunksAnalyzer` primary-ineligible (advisory-only) removes its 8 heldout
CLEAN-FPs (and 2 probe FPs) with no recall loss, because it has no measurable true-positive.

## Change (one, in `decision_policy_support.split_candidates`)
Add `_PRIMARY_INELIGIBLE_ANALYZERS = frozenset({"InconsistentChunksAnalyzer"})`. In the candidate
loop, suppress any candidate from this set (retained for traceability via `suppressed_reason=
"primary_ineligible_uncalibrated_zero_tp_analyzer"`, never eligible for primary). When it is the
lone fail candidate, the existing "no eligible candidate → CLEAN" path returns CLEAN.

## Hard acceptance criteria (revert on ANY fail)
1. Protected baseline check: pass (43/46, expected failures + category passes unchanged).
2. Calib native 23/45 unchanged; Calib default unchanged.
3. Real heldout native CLEAN-FP DOWN by ~8 (38→~30; overall up accordingly); NO gold-FAIL row
   newly mis-served; no NEW failure-type FP introduced on the previously-correct CLEAN rows.
4. Probe native ≥ prior (the 3 INCONSISTENT FP rows should now resolve toward CLEAN/their gold).
5. Decision-policy + analyzer test suites green (modulo known pre-existing stale fails).

## Guarded direction
The dangerous direction is suppressing a real failure → CLEAN. Mitigated because (a) the analyzer
has no measured TP to suppress, and (b) suppression is primary-eligibility only — the signal is
retained in the trace and still available to corroborate other analyzers. If any protected/Calib
TP turns out to depend on it, REVERT and instead scope to the `[profile] cX<->cY` sub-signal.
