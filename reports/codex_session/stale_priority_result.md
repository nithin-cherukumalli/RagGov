# Stale Priority Result

## Scope

Task 2 changed only primary-failure decision policy ordering. No analyzer predicates,
golden labels, protected acceptable alternatives, thresholds, gates, or production
gating settings were changed.

## Rationale

When `STALE_RETRIEVAL` and `INSUFFICIENT_CONTEXT` both fire at `status="fail"`,
structured stale-context evidence is more specific than generic retrieval
insufficiency. It identifies that the retrieved evidence is present but the wrong
version, while insufficiency only says the context is not adequate.

The active policy path is `decision_policy_support.candidate_sort_key`. The
specificity rank for `SufficiencyAnalyzer` stale-context evidence with
`[sufficiency:stale_context_mistaken_as_sufficient]` now ranks at `91`, just
above `RetrievalDiagnosisAnalyzerV0` `INSUFFICIENT_CONTEXT` at `90`.

The plain `StaleRetrievalAnalyzer` signal remains heuristic/advisory when it has
no structured version-validity report, preserving visible degradation behavior.

## Results

- Calib-50 case `govrag-calib-seed-038`: `INSUFFICIENT_CONTEXT` -> `STALE_RETRIEVAL`
- Calib-50 primary accuracy: `0.620`
- Calib-50 stage accuracy: `0.540`
- Calib-50 safety counters: `false_clean_count=0`, `false_security_count=0`, `false_incomplete_count=0`
- Heldout primary accuracy: `0.733`
- Heldout primary changes versus parser-validation baseline: `0`
- Protected baseline check: pass (`41/46` in current harness baseline)

## Files

- `src/raggov/decision_policy_support.py`
- `tests/decision_policy/test_primary_failure_policy.py`

