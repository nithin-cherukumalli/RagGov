# GovRAG-Calib Evaluation Report

- Dataset: `evals/govrag_calib/calib_150_seed.jsonl`
- Mode: `native`
- Total cases: `50`

## Calibration Status

- `calibration_status`: `not_calibrated`
- `production_gating_eligible`: `False`
- `confidence_intervals_available`: `False`
- `heldout_split_locked`: `False`

## Prediction Metrics

- `primary_failure_accuracy`: `0.48`
- `stage_accuracy`: `0.42`
- `first_failing_node_accuracy`: `unavailable`
- `fix_category_accuracy`: `0.0`
- `root_cause_accuracy`: `0.2`

## Safety Metrics

- `false_clean_count`: `0`
- `false_security_count`: `0`
- `false_incomplete_count`: `0`
- `dangerous_clean_miss_count`: `0`
- `security_stage_miss_count`: `0`
- `acceptable_nonclean_human_review_count`: `2`
- `dangerous_miss_count`: `0`
- `dangerous_miss_count_definition`: Deprecated alias for dangerous_clean_miss_count: a security/privacy/adversarial/high-risk case returned CLEAN or failed to require human review.
- `human_review_miss_count`: `1`

## Family Metrics

| Family | Total | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer_quality | 5 | 0.200 | 0.667 | 0.400 | 0.500 |
| citation | 7 | 0.429 | 0.444 | 0.571 | 0.500 |
| clean_pass | 5 | 0.400 | 1.000 | 0.400 | 0.571 |
| grounding | 8 | 0.750 | 0.300 | 0.750 | 0.429 |
| retrieval | 8 | 0.375 | 0.667 | 0.250 | 0.364 |
| security_privacy | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| sufficiency | 5 | 0.200 | 0.200 | 0.200 | 0.200 |
| version_validity | 5 | 0.200 | 1.000 | 0.200 | 0.333 |

## Decision Policy Metrics

- `expected_failure_candidate_generated_count`: `25`
- `expected_candidate_selected_count`: `24`
- `acceptable_alternative_match_count`: `9`
- `wrong_stage_correct_failure_count`: `5`
- `correct_stage_wrong_failure_count`: `2`

## Evidence Diagnostics

### Evidence Gap Flags

| Flag | Count |
| --- | ---: |
| `no_citations` | 38 |
| `unsupported_optional_metric` | 50 |

### Reasons Not Scored

| Reason | Count |
| --- | ---: |
| `diagnosis_extracted_no_claims` | 5 |
| `first_failing_node_label_unavailable` | 50 |

_This report is evaluation-only. It does not claim calibrated confidence or production gating eligibility._
