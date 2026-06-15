# GovRAG-Calib Evaluation Report

- Dataset: `evals/govrag_calib/splits/heldout_v0_1.jsonl`
- Mode: `native`
- Total cases: `15`

## Calibration Status

- `calibration_status`: `not_calibrated`
- `production_gating_eligible`: `False`
- `confidence_intervals_available`: `False`
- `heldout_split_locked`: `False`

## Prediction Metrics

- `primary_failure_accuracy`: `0.7333333333333333`
- `stage_accuracy`: `0.5333333333333333`
- `first_failing_node_accuracy`: `unavailable`
- `fix_category_accuracy`: `0.0`
- `root_cause_accuracy`: `0.15384615384615385`

## Safety Metrics

- `false_clean_count`: `0`
- `false_security_count`: `0`
- `false_incomplete_count`: `0`
- `dangerous_clean_miss_count`: `0`
- `security_stage_miss_count`: `0`
- `acceptable_nonclean_human_review_count`: `0`
- `dangerous_miss_count`: `0`
- `dangerous_miss_count_definition`: Deprecated alias for dangerous_clean_miss_count: a security/privacy/adversarial/high-risk case returned CLEAN or failed to require human review.
- `human_review_miss_count`: `0`

## Family Metrics

| Family | Total | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer_quality | 2 | 0.000 | 1.000 | 0.500 | 0.667 |
| citation | 2 | 1.000 | 0.667 | 1.000 | 0.800 |
| clean_pass | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| grounding | 2 | 1.000 | 0.500 | 1.000 | 0.667 |
| retrieval | 2 | 0.500 | 1.000 | 0.500 | 0.667 |
| security_privacy | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| sufficiency | 2 | 0.500 | 1.000 | 0.500 | 0.667 |
| version_validity | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

## Decision Policy Metrics

- `expected_failure_candidate_generated_count`: `9`
- `expected_candidate_selected_count`: `11`
- `acceptable_alternative_match_count`: `3`
- `wrong_stage_correct_failure_count`: `3`
- `correct_stage_wrong_failure_count`: `0`

## Evidence Diagnostics

### Evidence Gap Flags

| Flag | Count |
| --- | ---: |
| `no_citations` | 11 |
| `unsupported_optional_metric` | 15 |

### Reasons Not Scored

| Reason | Count |
| --- | ---: |
| `first_failing_node_label_unavailable` | 15 |

_This report is evaluation-only. It does not claim calibrated confidence or production gating eligibility._
