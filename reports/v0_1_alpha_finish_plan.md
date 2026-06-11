# v0.1 Alpha Finish Plan

## Launch Readiness Blockers

- `python scripts/launch_readiness.py || true`: generated Status `Not Ready`.
- Hard blockers currently reported: full pytest, external regression, decision-policy regression, common/subtle benchmark behavior, and previously surfaced regression cases.
- Production-only blockers: external provider runtime degradation and incomplete calibration. These must remain visible, but should not block alpha if native/common safety gates are intact.
- Current protected truth still holds: common native `41/46`, common external-enhanced `41/46`, false clean/security/incomplete all `0`, production gating `false`.

## Top Pytest Failure Clusters

- Claim external adapter contract:
  - `tests/evaluators/claim/test_refchecker_adapter.py::test_claim_grounding_records_refchecker_provider_in_evidence`
  - `tests/evaluators/claim/test_structured_llm_claim_verifier.py::*`
  - `tests/stresslab/evidence_layer/test_external_signal_provenance.py::test_claim_external_signal_provenance_is_preserved`
  - Likely files: `src/raggov/analyzers/grounding/support.py`, `src/raggov/evaluators/claim/structured_llm.py`
- Citation support stealing clean/retrieval classifications:
  - external regression clean/retrieval-noise tests
  - pinpointing retrieval coverage/precision tests
  - Likely files: decision/pinpointing or citation/grounding rollup path; inspect before editing.
- Subtle benchmark mismatches:
  - broad expected primary/stage mismatches in `tests/stresslab/test_subtle_benchmark_regressions_real.py`
  - Treat as RC/advisory unless false-clean or critical security/incomplete safety counters regress.
- Evidence-layer and triplet-verification tests:
  - claim type detection and triplet aggregation failures
  - Likely stale or secondary source issues; not first alpha blocker if common benchmark and grounding tests remain stable.
- Environment/provider degradation:
  - DeepEval/RAGAS degraded, RefChecker/RAGChecker unavailable.
  - This is production/RC-adjacent, not alpha blocking if degradation metadata is visible.

## Real Core Blockers

- Fix adapter crashes and missing metadata propagation. These are real source regressions because optional external providers should fail visibly or provide advisory signals without crashing grounding.
- Fix clean/retrieval cases being reclassified as citation support failures if caused by source behavior. This affects alpha honesty because launch readiness reports false-clean/security drift risks from the external regression suite.

## Stale Expectation Tests

- Some common/subtle exact-label tests expect old stage or root-cause choices while current common benchmark protected baseline accepts `41/46`.
- Do not update any golden labels or benchmark expected labels in this task.
- Only update stale unit expectations if a source behavior is clearly correct and the assertion still tests a useful invariant.

## Experimental/Advisory Failures

- Subtle suite broad mismatch rate.
- Full external-provider runtime availability.
- Production calibration and confidence intervals.
- These should remain visible as warnings or RC/production blockers, not alpha blockers.

## Exact Minimal Fix Sequence

1. Run required audit-only preflight because source/launch-readiness edits may affect protected behavior.
2. Patch claim adapter contract regressions:
   - allow injected extractor clients that already implement `extract_structured`
   - include `support_label` when `StructuredLLMClaimVerifierAdapter.verify()` returns `VerificationResult`
3. Run targeted adapter/provenance tests.
4. Investigate citation support stealing clean/retrieval classifications using the smallest failing stresslab tests.
5. If source behavior is wrong, patch the narrow decision/pinpointing path and run focused stresslab external/pinpointing tests.
6. Adjust launch readiness classification only if after source fixes it still treats documented alpha non-goals as hard blockers:
   - production calibration: production blocker
   - degraded external providers: advisory/production blocker
   - subtle suite exact-label weakness: RC blocker unless false-clean/security/incomplete counters regress
   - full pytest: RC blocker unless failures include alpha safety regressions
7. Validate common benchmark, harness, decision policy, grounding, launch readiness, and full pytest summary.

## Files Likely Needing Edits

- `src/raggov/analyzers/grounding/support.py`
- `src/raggov/evaluators/claim/structured_llm.py`
- Possibly `stresslab/runners/launch_readiness.py`
- Possibly focused tests under `tests/evaluators/claim/` or `tests/stresslab/test_launch_readiness.py` if launch-readiness semantics change.
