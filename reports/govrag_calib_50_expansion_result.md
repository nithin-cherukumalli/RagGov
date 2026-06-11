# GovRAG-Calib 50 Expansion Result

## Final Recommendation

`GOVRAG_CALIB_50_READY`

## Files Changed

- `evals/govrag_calib/calib_150_seed.jsonl`
- `evals/govrag_calib/results/govrag_calib_eval_latest.json`
- `evals/govrag_calib/results/govrag_calib_eval_latest.md`
- `reports/govrag_calib_50_expansion_result.md`

No analyzer logic, decision policy, engine behavior, benchmark labels, golden fixtures, thresholds, alpha release gates, or production gating settings were changed.

## Case Count

- Before: `10`
- After: `50`
- Added reviewed records: `40`
- `calibration_split`: all `unset`
- `calibration_status`: `10 seed`, `40 reviewed`
- `heldout_locked`: `0`

## Failure Family Distribution

| Failure family | Cases | Required minimum for this step | GovRAG-Calib-150 target |
| --- | ---: | ---: | ---: |
| `clean_pass` | 5 | 5 | 15 |
| `retrieval` | 8 | 8 | 25 |
| `grounding` | 8 | 8 | 25 |
| `citation` | 7 | 7 | 20 |
| `sufficiency` | 5 | 5 | 15 |
| `version_validity` | 5 | 5 | 15 |
| `security_privacy` | 7 | 7 | 20 |
| `answer_quality` | 5 | 5 | 15 |

Additional quality coverage:

- Multi-document retrieval cases: `32`
- Cases with explicit citations: `12`
- Cases with metadata/freshness/version requirements or chunk metadata: `50`
- Adversarial or security-relevant cases: `9`
- Generation-stage cases: `5`

## Validation Result

`python scripts/validate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl`: passed.

Expected warnings remain because the dataset is below the 150-case target and every family is still below its final GovRAG-Calib-150 target.

## Evaluator Result

`python scripts/evaluate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl --mode native`: passed and regenerated JSON plus Markdown outputs.

Metrics from `evals/govrag_calib/results/govrag_calib_eval_latest.json`:

- Total cases: `50`
- `primary_failure_accuracy`: `0.42`
- `stage_accuracy`: `0.4`
- `first_failing_node_accuracy`: `unavailable`
- `fix_category_accuracy`: `0.0`
- `root_cause_accuracy`: `0.2222222222222222`
- `false_clean_count`: `7`
- `false_security_count`: `0`
- `false_incomplete_count`: `0`
- `dangerous_miss_count`: `3`
- `human_review_miss_count`: `8`
- `acceptable_alternative_match_count`: `4`
- `expected_failure_candidate_generated_count`: `22`
- `expected_candidate_selected_count`: `21`

Calibration status remains:

- `calibration_status`: `not_calibrated`
- `production_gating_eligible`: `false`
- `confidence_intervals_available`: `false`
- `heldout_split_locked`: `false`

## Weak Families

The expanded scaffold is useful for surfacing evaluator gaps, not for claiming calibrated quality.

Weakest per-family primary accuracies:

- `answer_quality`: `0.2`
- `sufficiency`: `0.2`
- `version_validity`: `0.2`
- `citation`: `0.2857142857142857`
- `retrieval`: `0.375`
- `clean_pass`: `0.4`

All families remain below the final 150-case distribution target.

## Alpha Release Gate

`python scripts/check_v0_1_alpha_release.py`: passed.

Protected alpha truth remains:

- common benchmark native: `41/46`
- common benchmark external-enhanced: `41/46`
- false clean/security/incomplete counts: `0/0/0`
- launch readiness: `v0.1-alpha-clean Ready`
- production gating remains disabled
- production remains `Not Ready`

## Why This Is Not Production Calibration

This expansion creates reviewed seed/review cases only. There is no heldout split, no locked heldout labels, no adjudicated calibration set, no confidence interval computation, and no production gating eligibility. The evaluator reports diagnostic agreement metrics on scaffold data, not calibrated confidence.

## Next Step Toward 150

Expand from 50 to 150 cases with adjudication workflow and split assignment, then lock a heldout subset before any confidence calibration or production-gating discussion.
