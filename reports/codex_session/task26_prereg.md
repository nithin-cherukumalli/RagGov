# Task 26 pre-registration — CLEAN→INSUFFICIENT_CONTEXT over-firing (scope condition)

**Date:** 2026-06-18 · written BEFORE code.

## Root cause (instrumented)
4 of the 5 `CLEAN→INSUFFICIENT_CONTEXT` probe false positives fire via SufficiencyAnalyzer Pattern 5
(`missing_scope_condition`, `sufficiency.py` ~1066): it fires whenever a US-state name appears in any
retrieved chunk (`_SCOPE_PREP_PATTERN`) and is absent from the query — regardless of whether the
query is actually scope-general. On specific multi-hop lookups ("Gunmen from Laredo starred which
narrator of Frontier?"), an incidental "Texas" triggers a spurious insufficiency though the answer
("Walter Darwin Coy") is correct. (Row 5 is a different term-coverage path — out of scope.)

## Discriminator
Genuine case (unit test `test_sufficiency_missing_scope_emits_missing_scope_condition_marker`):
"What is the sales tax?" + "Sales tax in California is 7.25%" — a scope-GENERAL query (no
disambiguating proper-noun entity) whose answer is inherently location-dependent. The FP queries are
specific entity lookups containing proper nouns (Laredo/Frontier, Zilpo Road, Ryoichi Ikegami…).
Prototype: a "query has a disambiguating proper-noun entity" gate suppresses all 4 FPs and keeps
both genuine unit-test queries (0 proper nouns).

## Change (one, narrow)
In SufficiencyAnalyzer Pattern 5, only emit `missing_scope_condition` when the query is scope-general
— i.e. it contains no disambiguating proper-noun entity (capitalized non-leading, non-interrogative
token). All other patterns and markers unchanged.

## Hard acceptance criteria
1. Protected baseline stays **43/46** (check pass). *(revert)*
2. Calib scored primary stays **≥ 23/45** (scope marker never fires on Calib today). *(revert)*
3. Probe overall stays **≥ 80/145**; no genuine INSUFFICIENT_CONTEXT true positive flips away. *(revert)*
4. `test_sufficiency_regression.py` + `tests/decision_policy/test_primary_failure_policy.py` +
   `test_analyzers`/`decision_policy` suites green (modulo pre-existing stale fail). *(revert)*

**Success:** `CLEAN→INSUFFICIENT_CONTEXT` (scope path) drops 4 → ≤1; CLEAN-correct rises.
