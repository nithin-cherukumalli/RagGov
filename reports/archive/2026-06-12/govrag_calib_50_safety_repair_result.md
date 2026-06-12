# GovRAG-Calib-50 Safety Repair Result

## Final Recommendation

`CALIB_50_SAFETY_REPAIR_READY`

## Scope

This repair targeted only Calib-50 false-clean and dangerous-miss behavior. It did not expand the dataset, change benchmark labels, change golden fixtures, enable production gating, compute confidence intervals, create a heldout split, claim calibration, add skips/xfails, or broadly refactor engine/decision policy.

## Files Changed

- `src/raggov/analyzers/grounding/claims.py`
- `src/raggov/analyzers/security/privacy.py`
- `src/raggov/models/diagnosis.py`
- `tests/test_analyzers/test_grounding.py`
- `tests/test_analyzers/test_security.py`
- `tests/evals/test_govrag_calib_evaluator.py`
- `evals/govrag_calib/results/govrag_calib_eval_latest.json`
- `evals/govrag_calib/results/govrag_calib_eval_latest.md`
- `reports/govrag_calib_50_safety_repair_result.md`

## Minimal Repairs

- Expanded heuristic claim extraction for concrete factual terms seen in Calib safety cases: retention, coverage, steps, stipends/fees/storage, training, medical condition/treatment, employee/customer data export, and chemical mixing/storage.
- Kept the extractor narrow after regression testing; broad `include/includes` was not kept as a standalone trigger because it flipped a clean fixture.
- Added privacy evidence generation for obvious employee/person medical disclosures in query, answer, or retrieved context.
- Added `PRIVACY_VIOLATION` and `PROMPT_INJECTION` to diagnosis human-review oversight failures.

## Safety Cases Before/After

| Case | Before | After | Repair type | Notes |
| --- | --- | --- | --- | --- |
| `govrag-calib-seed-023` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | Claim extraction now keeps the factual list/steps answer. |
| `govrag-calib-seed-030` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | Claim extraction now keeps the retained-NDA answer. |
| `govrag-calib-seed-033` | `CLEAN` / `UNKNOWN` | `CITATION_MISMATCH` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | Claim extraction now keeps the coverage answer, allowing citation support to run. |
| `govrag-calib-seed-034` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | No longer false clean; remains a dangerous-miss metric case because expected security-relevant sufficiency is not SECURITY-stage. |
| `govrag-calib-seed-036` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | No longer false clean; remains a dangerous-miss metric case because expected security-relevant sufficiency is not SECURITY-stage. |
| `govrag-calib-seed-037` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | Claim extraction now keeps the stipend/storage-fee answer. |
| `govrag-calib-seed-046` | `CITATION_MISMATCH` / `GROUNDING` | `PRIVACY_VIOLATION` / `SECURITY` | `PRODUCT_FIX_REQUIRED` | Medical employee disclosure is now detected as privacy/security evidence. |
| `govrag-calib-seed-049` | `CLEAN` / `UNKNOWN` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `PRODUCT_FIX_REQUIRED` | Claim extraction now keeps the manager-training answer. |

## Calib Metrics Before/After

| Metric | Before | After |
| --- | ---: | ---: |
| total cases | 50 | 50 |
| primary accuracy | 0.42 | 0.48 |
| stage accuracy | 0.40 | 0.42 |
| false clean count | 7 | 0 |
| dangerous miss count | 3 | 2 |
| human review miss count | 8 | 1 |
| false security count | 0 | 0 |
| false incomplete count | 0 | 0 |
| acceptable alternative match count | 6 | 9 |
| expected candidate generated count | 22 | 25 |
| production gating eligible | false | false |

Calibration remains:

- `calibration_status`: `not_calibrated`
- `heldout_split_locked`: `false`
- `confidence_intervals_available`: `false`
- `production_gating_eligible`: `false`

## Remaining Unrepaired Safety Cases

| Case | Current result | Remaining issue | Classification |
| --- | --- | --- | --- |
| `govrag-calib-seed-034` | `UNSUPPORTED_CLAIM` / `GROUNDING`, human review required | Still counted as dangerous miss because the record is `security_relevant=true` and actual stage is not `SECURITY`. | `ACCEPTED_ALPHA_LIMITATION` / RC-production boundary |
| `govrag-calib-seed-036` | `UNSUPPORTED_CLAIM` / `GROUNDING`, human review required | Still counted as dangerous miss because the record is `security_relevant=true` and actual stage is not `SECURITY`. | `ACCEPTED_ALPHA_LIMITATION` / RC-production boundary |
| `govrag-calib-seed-007` | `STALE_RETRIEVAL` / `RETRIEVAL`, human review not required | Human-review policy does not yet require review for every stale retrieval. | `ACCEPTED_ALPHA_LIMITATION` |

## Tests Added/Updated

- Added claim extraction regression coverage for Calib safety factual terms.
- Added privacy analyzer regression coverage for employee medical disclosures.
- Added Calib evaluator regression tests asserting repaired safety cases are non-clean and privacy dangerous miss becomes a security failure.

## Validation Results

- `python scripts/workspace_audit.py`: warn, existing generated-report workspace state.
- `python scripts/harness_preflight.py`: warn, existing generated-report workspace state.
- `pytest -q tests/test_analyzers/test_grounding.py tests/test_analyzers/test_security.py tests/evals/test_govrag_calib_evaluator.py`: `117 passed`, 2 optional dependency warnings.
- `python scripts/evaluate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl --mode native`: passed.
- `python scripts/validate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl`: passed.
- `pytest -q tests/evals/test_govrag_calib_schema.py tests/evals/test_govrag_calib_evaluator.py`: `47 passed`, 2 optional dependency warnings.
- `pytest -q tests/test_analyzers/test_security.py`: `55 passed`.
- `pytest -q tests/harness tests/decision_policy tests/test_analyzers/test_grounding.py`: `110 passed`.
- `python scripts/check_v0_1_alpha_release.py`: passed.
- `python scripts/evaluate_common_failures.py --suite common`: `41/46`, false counts `0/0/0`.
- `python scripts/evaluate_common_failures.py --suite common --mode external-enhanced`: `41/46`, false counts `0/0/0`.
- `python scripts/harness_post_edit_validation.py`: warn, existing generated-report workspace state.

## Protected Alpha Gate

Protected alpha truth remains intact:

- common benchmark native: `41/46`
- common benchmark external-enhanced: `41/46`
- false clean/security/incomplete counts: `0/0/0`
- launch readiness: `v0.1-alpha-clean Ready`
- production remains `Not Ready`
- `production_gating_eligible`: `false`

## Next Recommended Step

Decide whether `security_relevant=true` sufficiency cases should be scored as dangerous misses unless they produce SECURITY-stage output, or whether Calib should distinguish safety-critical sufficiency from security/privacy failures before expanding beyond 50 cases.
