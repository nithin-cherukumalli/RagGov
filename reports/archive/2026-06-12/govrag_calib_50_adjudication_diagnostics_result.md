# GovRAG-Calib-50 Adjudication Diagnostics Result

## Final Recommendation

`CALIB_50_ADJUDICATION_READY`

## Scope

This pass adjudicated the five Calib-50 label/schema issue candidates and added evaluator-only evidence diagnostics. No analyzer logic, decision policy, engine behavior, benchmark labels, golden fixtures, thresholds, alpha release gates, heldout split, confidence intervals, or production gating settings were changed.

## Label/Schema Candidates Reviewed

| Case | Triage issue | Decision | Dataset change |
| --- | --- | --- | --- |
| `govrag-calib-seed-010` | LOW_CONFIDENCE case can also trigger native citation-support diagnosis. | `ADD_ACCEPTABLE_ALTERNATIVE` | Added `CITATION_MISMATCH`; kept primary label and human-review expectation. |
| `govrag-calib-seed-011` | Clean-pass case flagged as citation mismatch by native evaluator. | `KEEP_LABEL` | No label change; evidence supports the clean answer and citation. |
| `govrag-calib-seed-012` | Clean-pass case can be interpreted as retrieval noise. | `MARK_REVIEW_REQUIRED` | Kept `CLEAN`; changed `label_status` from `reviewed` to `seed`; updated notes/provenance. |
| `govrag-calib-seed-013` | Clean-pass case flagged as citation mismatch by native evaluator. | `KEEP_LABEL` | No label change; evidence supports the clean answer and citation. |
| `govrag-calib-seed-050` | LOW_CONFIDENCE case can also trigger metadata/scope issue. | `ADD_ACCEPTABLE_ALTERNATIVE` and `MARK_REVIEW_REQUIRED` | Added `METADATA_LOSS`; changed `label_status` from `reviewed` to `seed`; updated notes/provenance. |

## Dataset Fields Changed

- `govrag-calib-seed-010`: added `CITATION_MISMATCH` to `acceptable_alternative_failures`; updated notes/provenance.
- `govrag-calib-seed-012`: changed `label_status` from `reviewed` to `seed`; updated notes/provenance.
- `govrag-calib-seed-050`: added `METADATA_LOSS` to `acceptable_alternative_failures`; changed `label_status` from `reviewed` to `seed`; updated notes/provenance.

Dataset remains exactly `50` cases.

Label status after adjudication:

- `seed`: `12`
- `reviewed`: `38`

Calibration status remains unchanged:

- `seed`: `10`
- `reviewed`: `40`
- no `heldout_locked` cases

## Labels Changed

No `expected_primary_failure`, `expected_stage`, `failure_family`, or `expected_human_review_required` labels were changed.

## Acceptable Alternatives Added

- `govrag-calib-seed-010`: `CITATION_MISMATCH`
- `govrag-calib-seed-050`: `METADATA_LOSS`

## Cases Marked Review-Required

No case was moved to a heldout or adjudicated status. The following cases were downgraded to `label_status: seed` because uncertainty remains:

- `govrag-calib-seed-012`
- `govrag-calib-seed-050`

## Evaluator Diagnostics Added

Each per-case evaluator row now includes `evidence_diagnostics`:

- `expected_claim_count`
- `extracted_claim_count`
- `skipped_claim_count`
- `citation_count`
- `cited_doc_ids`
- `missing_expected_claim_ids`
- `missing_expected_doc_ids`
- `diagnosis_has_claim_evidence`
- `diagnosis_has_citation_evidence`
- `expected_candidate_generated`
- `reason_not_scored`
- `evidence_gap_flags`

The evaluator also emits `evidence_diagnostics_summary` with evidence-gap flag counts and not-scored reason counts. These diagnostics do not affect scoring.

Current diagnostics summary:

- `no_citations`: `38`
- `unsupported_optional_metric`: `50`
- `diagnosis_extracted_no_claims`: `13`
- `first_failing_node_label_unavailable`: `50`

## Validator Warnings Added

The validator now warns, without failing, for seed/review evidence gaps:

- non-clean cases with empty `expected_affected_claim_ids`
- citation/retrieval/grounding/sufficiency/version-validity cases with retrieved chunks but empty `expected_affected_doc_ids`
- citation-family cases with no citations
- version/staleness cases with empty `metadata_requirements`
- high-risk security/adversarial cases that do not require human review
- ambiguous retrieval/sufficiency/grounding/answer-quality cases with no acceptable alternatives

The current 50-case dataset passes validation and triggers only the existing below-150 distribution warnings.

## Metrics Before/After

| Metric | Before | After |
| --- | ---: | ---: |
| total cases | 50 | 50 |
| primary failure accuracy | 0.42 | 0.42 |
| stage accuracy | 0.40 | 0.40 |
| acceptable alternative match count | 4 | 6 |
| false clean count | 7 | 7 |
| dangerous miss count | 3 | 3 |
| human review miss count | 8 | 8 |
| production gating eligible | false | false |

The primary and stage metrics stayed unchanged because no primary/stage labels were changed. The acceptable-alternative count increased from `4` to `6`, reflecting conservative adjudication of underspecified seed cases.

## Validation Commands

- `python scripts/workspace_audit.py`: warn, generated-report workspace state.
- `python scripts/harness_preflight.py`: warn, generated-report workspace state.
- `pytest -q tests/evals/test_govrag_calib_schema.py tests/evals/test_govrag_calib_evaluator.py`: `45 passed`, 2 optional dependency warnings.
- `python scripts/validate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl`: passed.
- `python scripts/evaluate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl --mode native`: passed.
- `python scripts/check_v0_1_alpha_release.py`: passed.

## Alpha Gate Result

Alpha release gate passed:

- common benchmark native: `41/46`
- common benchmark external-enhanced: `41/46`
- false clean/security/incomplete counts: `0/0/0`
- launch readiness: `v0.1-alpha-clean Ready`
- production gating remains disabled
- production remains `Not Ready`

## Why Calibration Remains Not Calibrated

This pass did not create a heldout split, adjudicated calibration set, confidence intervals, or production-gating policy. The dataset remains seed/review scaffolding, and the evaluator reports diagnostics only.

## Next Recommended Step

Use the new evidence diagnostics to repair or adjudicate the remaining false-clean and dangerous-miss ownership buckets before expanding beyond 50 cases.
