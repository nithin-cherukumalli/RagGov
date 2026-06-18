# Task 23 pre-registration — source-assertion false-CLEAN repair

**Date:** 2026-06-18 · written BEFORE code. Implements the Codex sidekick report's
recommended first false-CLEAN patch ("Source-Assertion False-CLEAN Repair").

## Baseline reconciliation (important)

The sidekick ledger measured this on an earlier engine state (pre Tasks 19–21) and reported
`UNSUPPORTED_CLAIM → CLEAN = 15`, false-CLEAN total 24. **Re-measured now (post Tasks 18–21,
my scorer):**
- Probe overall: 57/145 = 0.393.
- False-CLEAN total = **15**: `UNSUPPORTED_CLAIM→CLEAN` **7**, `INSUFFICIENT_CONTEXT→CLEAN` **6**,
  `CONTRADICTED_CLAIM→CLEAN` **2**.
- Expected-CLEAN false positives = 17 (CLEAN-correct 13/30).

So the mechanism is identical but the magnitude is smaller. Targets below use the re-measured 7.

## Root cause (confirmed by instrumentation)

All 7 `UNSUPPORTED_CLAIM→CLEAN` rows carry the same fabricated suffix sentence:
`"The source also notes this was formally reaffirmed at a later international summit."`
`HeuristicClaimExtractorV0` marks it `should_verify=False, skip_reason="lacks_substantive_terms"`
because it contains no number/date and none of `_SUBSTANTIVE_RE`'s verbs. With every claim
skipped, `ClaimGroundingAnalyzer` cannot fail and the engine returns CLEAN — a dangerous miss
(real unsupported content reported as clean).

## Change (one, narrow)

Add `_SOURCE_ASSERTION_RE` to `src/raggov/analyzers/grounding/claims.py`: a source/passage/
document/text/report/author subject + assertion verb (notes/states/reports/confirms/documents/
says/mentions/indicates/asserts/claims/observes/reaffirm(s|ed)). When a sentence matches, treat
it as substantive (`should_verify=True`), mirroring the existing `_SHORT_ENTITY_RE` /
`_ENTITY_ATTRIBUTE_CLAIM_RE` hooks. No threshold/gate/policy change. List/short-answer recall is
explicitly **out of scope** (higher CLEAN-precision risk; deferred to Task 24).

## Hard acceptance criteria

1. Protected baseline stays **41/46 GREEN**. *(revert trigger)*
2. Calib scored primary stays **≥ 23/45 = 0.511**. *(revert trigger)*
3. Probe overall accuracy does **not decrease** (≥ 57/145). *(revert trigger)*
4. Expected-CLEAN false positives do **not rise above 17** (CLEAN-correct stays ≥ 13/30).
   *(revert trigger)*

**Success criterion:**
5. `UNSUPPORTED_CLAIM → CLEAN` drops **7 → ≤2**.

If the suffix becomes verifiable but the native heuristic verifier still passes/skips it (so the
row stays CLEAN), that is a verifier/selection issue — document it and do not broaden policy in
this task. Revert if any of 1–4 fails.

## Tests
- `tests/test_analyzers/test_claim_extractor.py`: source-assertion suffix → verifiable; short
  non-substantive text still skipped; generic explanatory prose without anchors not over-promoted.
- `tests/test_analyzers/test_grounding.py`: a substantive factual answer whose only unsupported
  content is a source-assertion suffix must not silently diagnose CLEAN.
