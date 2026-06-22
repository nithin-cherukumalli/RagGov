# Phase 2 increment 4 — grounded-clean gate v2 (entailed-fraction, not strict-zero)

**Date:** 2026-06-22 · written BEFORE code. Supersedes the strict-zero rule of increment 3.

## Why increment 3 didn't move the number
The Kimi A/B (with increment 3 live) showed NO gain: native 18/75 (CLEAN-FP 0.76) vs
Kimi-NLI 16/75 (CLEAN-FP 0.78). The gate barely fired because its condition 3 was
**strict-zero**: it required the entailment verdict to find *zero* `unsupported` claims.
On real faithful answers NLI almost always flags *some* peripheral claim `unsupported`
(or extracts none), so the strict rule never triggered. The gate was correct but too
narrow. (Caveat: that A/B was also 42% rate-limit fallback — Step B / commit ee0b433
addresses the reliability half; this prereg addresses the strictness half.)

## Change (one, in `decision_policy_support`)
Replace the strict-zero `_answer_is_entailment_clean` with an **entailed-fraction** rule.
Partition the ClaimGroundingAnalyzer claim labels:
- positive  = {entailed, supported}
- soft-neg  = {unsupported}
- hard-neg  = {contradicted}
- abstain   = {insufficient_evidence, abstain, skipped, unverifiable}  ← excluded from denominator

Let `verifiable = positive + soft-neg + hard-neg`. Suppress the low-tier retrieval-health
winner iff ALL of:
1. an entailment-grade `verification_method` was used (unchanged gate to native-safety); AND
2. **no `contradicted` claim** (hard floor — a real contradiction is never suppressed); AND
3. `verifiable > 0` (require positive grounding evidence — tightens the "extracts none"
   case where the old rule wrongly fired on all-abstain); AND
4. `positive / verifiable >= THRESHOLD` (default 0.75, overridable via env
   `RAGGOV_GROUNDED_CLEAN_ENTAILED_FRACTION` so S1 can sweep without code edits).

Suppressible failure set and the no-BLOCKING_DETERMINISTIC guard are **unchanged** from
increment 3.

## INCOMPLETE_DIAGNOSIS — deliberately NOT added (decision recorded)
The prior NLI run produced several false primaries of `INCOMPLETE_DIAGNOSIS`, which is not
in the suppressible set. We do **not** add it here. INCOMPLETE_DIAGNOSIS is a meta/decision
failure ("could not reach a confident diagnosis"), not a retrieval-health heuristic; an
entailment-clean grounding verdict is not sufficient evidence that the *diagnosis* is
complete (it can co-occur with uncovered security/scope concerns). Adding it is a separate,
higher-risk change that needs its own prereg. Thinking-before-adding, per discipline.

## Why safe (no-regression argument)
Condition 1 is unchanged: in default/native mode no claim carries an entailment-grade
method, so the gate returns False and native behavior is byte-identical. Dangerous
direction (real failure → CLEAN) is guarded three ways: BLOCKING_DETERMINISTIC short-circuit,
the `contradicted` hard floor, and the requirement of real positive evidence (`verifiable>0`).
The only behavioral loosening vs increment 3 is tolerating a bounded fraction of `unsupported`
claims — and only for the six low-tier retrieval-health types, never for security/parser/
contradiction signals.

## Hard acceptance criteria (native — I verify these now)
1. Protected baseline check: pass. *(revert on fail)*
2. Calib native 23/45 unchanged. *(revert)*
3. Real heldout native 19/75 unchanged (native-mode, the byte-identical path). *(revert)*
4. `tests/` decision-policy + grounding suites green (modulo known pre-existing stale fails). *(revert)*

## Success (measured off-sandbox by the user/S1 with a CLEAN, post-backoff NLI run)
Real-heldout CLEAN-FP drops materially in NLI mode (target ≤ 0.60) with **zero** gold-FAIL
row (25 of them) flipping to CLEAN. Sweep THRESHOLD ∈ {0.6, 0.7, 0.75, 0.8, 0.9} via the env
override; pick the lowest CLEAN-FP that holds gold-FAIL recall flat. Unit tests prove:
entailed-fraction ≥ T + low-tier winner → CLEAN; one `contradicted` claim → NOT suppressed;
all-abstain → NOT suppressed.
