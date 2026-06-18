# Task 22 pre-registration — CONTRADICTED_CLAIM recall (native mode)

**Date:** 2026-06-18 · written BEFORE code.

## Hypothesis (from Codex sidekick audit)
Several expected `CONTRADICTED_CLAIM` probe rows emit contradiction-like native evidence
(`ClaimGroundingAnalyzer` fail+CONTRADICTED) but primary selection collapses to
`UNSUPPORTED_CLAIM`. Promoting genuine contradiction should raise recall (0/15).

## Hard acceptance criteria
1. Protected baseline stays 41/46. *(revert)*
2. Calib scored primary ≥ 23/45. *(revert)*
3. Probe overall ≥ 80/145; expected-CLEAN FPs not above 17. *(revert)*
4. No rise in false `CONTRADICTED_CLAIM` on non-contradiction rows. *(revert)*

**Success:** named CONTRADICTED rows (1,3,4 etc.) promote to CONTRADICTED_CLAIM.

## Scope decision
One of: (a) promotion/metadata fix, (b) verifier recall fix with structured evidence only,
(c) label-audit-only / no code if rows are noisy migrated labels. Native mode only; no
NLI/sentence-transformers dependency.
