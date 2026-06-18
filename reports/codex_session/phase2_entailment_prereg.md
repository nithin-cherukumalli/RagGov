# Phase 2 (entailment tier) — increment 1 pre-registration: entailment-grade contradiction is explicit

**Date:** 2026-06-18 · written BEFORE code. Master Plan Phase 2 (the trust lever).

## Hypothesis
A contradicted claim verified by an **entailment-grade method** (NLI / LLM entailment / structured-
LLM / conservative-ensemble) is hard semantic evidence and should be treated as an **explicit**
contradiction by the decision policy — so it promotes to `CONTRADICTED_CLAIM`. A native **lexical**
contradiction (HeuristicValueOverlapVerifier) stays demoted (the Task 22 no-go was correct: native
heuristic contradiction is not trustworthy). This realizes the hybrid tier: native stays
conservative; the optional NLI tier unlocks trustworthy contradiction.

## Change (one, narrow)
In `decision_policy_support.py::_has_explicit_contradiction`, add: a contradicted claim whose
`ClaimResult.verification_method` is in `_ENTAILMENT_GRADE_METHODS = {llm_claim_entailment_verifier_v1,
structured_llm_claim_verifier_v1, conservative_ensemble_v1}` counts as explicit. Nothing else changes.

## Why this is safe (no-regression argument)
In default/native mode (and in this sandbox) there is no `llm_client`, so the configured verifier is
the heuristic one; no claim gets an entailment `verification_method`; the new branch never fires →
default/native behavior is byte-for-byte unchanged. The change only activates when a real entailment
verifier is configured (the user's machine / a key / a future local-NLI model).

## Hard acceptance criteria
1. Protected baseline 43/46 unchanged. *(revert)*
2. Calib 23/45 unchanged. *(revert)*
3. Synthetic probe 80/145 (default) unchanged; real heldout v1 18/75 (default) unchanged. *(revert)*
4. `tests/test_analyzers` + `tests/decision_policy` green (modulo pre-existing stale fail). *(revert)*

**Success:** a unit test proves (a) an entailment-method contradicted claim → CONTRADICTED_CLAIM
promoted (explicit), and (b) a lexical-method contradicted claim → still demoted (not explicit).

This is the foundation; the real accuracy lift requires running with an actual entailment model
(out-of-sandbox, like the data pull) — that is a later increment, measured on the real heldout via
S1's NLI A/B harness.
