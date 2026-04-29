# Claim-Level Diagnostic Evaluation Report

- Evaluation status: `diagnostic_gold_v0_small_unvalidated`
- Case count: 10

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| claim_label_accuracy | 1.00 |
| citation_validity_accuracy | 1.00 |
| freshness_validity_accuracy | 1.00 |
| sufficiency_accuracy | 1.00 |
| a2p_primary_cause_accuracy | 1.00 |
| primary_stage_accuracy | 1.00 |
| fix_category_exact_accuracy | 1.00 |
| fix_category_partial_accuracy | 1.00 |

## Per-Case Summary
- `supported_1`: **PASS**
- `supported_2`: **PASS**
- `unsupported_missing_1`: **PASS**
- `unsupported_missing_2`: **PASS**
- `contradicted_1`: **PASS**
- `contradicted_2`: **PASS**
- `stale_source_case`: **PASS**
- `citation_mismatch_case`: **PASS**
- `weak_ambiguous_case`: **PASS**
- `insufficient_context_abstain_case`: **PASS**

## Mismatches
- None

## Raw Summary

```text
Claim-Level Diagnostic Evaluation Harness v0
evaluation_status=diagnostic_gold_v0_small_unvalidated
total_examples=10

Aggregate Metrics (by axis):
  Axis A - Claim Support:       1.00
  Axis B - Citation Validity:   1.00
  Axis C - Freshness Validity:  1.00
  Axis D - Context Sufficiency: 1.00
  Axis E - A2P Root Cause:      1.00
  Primary Stage:                1.00
  Fix Category (exact):         1.00
  Fix Category (partial):       1.00

Per-example results:
- supported_1: PASS
- supported_2: PASS
- unsupported_missing_1: PASS
- unsupported_missing_2: PASS
- contradicted_1: PASS
- contradicted_2: PASS
- stale_source_case: PASS
- citation_mismatch_case: PASS
- weak_ambiguous_case: PASS
- insufficient_context_abstain_case: PASS
```
