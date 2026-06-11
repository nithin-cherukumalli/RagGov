# GovRAG Launch Readiness

Status: **v0.1-alpha-clean Ready**

## Gates
- PASS: false_clean_count == 0
- PASS: false_incomplete_count == 0
- PASS: no external advisory signal becomes primary failure alone
- PASS: no retrieval anomaly becomes security without explicit security evidence
- PASS: prompt injection detection still passes
- PASS: privacy violation detection still passes
- PASS: provider missing/degraded reasons visible
- PASS: calibrated confidence absent unless calibration artifact exists
- PASS: every non-clean diagnosis has recommended fix
- PASS: every failed case has first_failing_node or explicit reason why unavailable
- FAIL: benchmark accuracy meets configured threshold
- FAIL: all previously surfaced regression cases pass
- FAIL: full pytest passes
- FAIL: common failure golden suite passes
- FAIL: subtle failure golden suite passes
- PASS: external regression suite passes
- PASS: no-silent-fallback checks pass
- PASS: decision policy regression checks pass
- PASS: native mode ignores missing optional external providers
- PASS: external-enhanced provider degradation is visible
- PASS: production gating remains disabled without calibration evidence
- PASS: v0.1-alpha common benchmark protected baseline holds
- PASS: v0.1-alpha safety gates pass

## Failure Reasons
- benchmark_accuracy=27% is below threshold 95%.
- Previously surfaced regression cases failed: retrieval_top_k_too_small_08, grounding_date_hallucination_20, security_retrieval_anomaly_only_36, quality_incomplete_38, quality_ignores_context_41, subtle_correct_unsupported_01, subtle_incomplete_answer_02, subtle_plausible_hallucination_03, subtle_related_non_supporting_04, subtle_answer_drift_06, subtle_ambiguous_query_07, subtle_local_contradiction_08, subtle_table_value_swap_11, subtle_near_miss_retrieval_13, subtle_constraint_override_14, subtle_many_weak_citations_15, full_pytest
- full pytest failed with exit_code=1. FAILED tests/test_analyzers/test_triplet_verification.py::test_triplet_verification_flow; FAILED tests/test_analyzers/test_triplet_verification.py::test_triplet_verification_fallback_on_extraction_failure; FAILED tests/test_analyzers/test_version_validity_pipeline.py::test_version_validity_decision_trace_explains_downstream_claim_failure; FAILED tests/test_pr5e_answer_quality.py::test_incomplete_answer_with_good_context_stage_generation; 36 failed, 1685 passed, 3 skipped, 31 warnings in 21.48s
- common_failure_suite failed at 89%. Mismatches: retrieval_top_k_too_small_08, grounding_date_hallucination_20, security_retrieval_anomaly_only_36, quality_incomplete_38, quality_ignores_context_41.
- subtle_failure_suite failed at 27%. Mismatches: subtle_correct_unsupported_01, subtle_incomplete_answer_02, subtle_plausible_hallucination_03, subtle_related_non_supporting_04, subtle_answer_drift_06, subtle_ambiguous_query_07, subtle_local_contradiction_08, subtle_table_value_swap_11, subtle_near_miss_retrieval_13, subtle_constraint_override_14, subtle_many_weak_citations_15.

## Launch Blockers
- `benchmark_behavior` (medium): benchmark_accuracy=27% below threshold 95% Remediation: Repair benchmark mismatches before release candidate readiness; alpha requires protected common baseline and zero false-clean/security/incomplete counters.
- `code_test_health` (medium): full pytest failed Remediation: Inspect captured check details and rerun this check after repair.
- `benchmark_behavior` (medium): common_failure_suite failed Remediation: Investigate failed benchmark cases and rerun the suite.
- `benchmark_behavior` (medium): subtle_failure_suite failed Remediation: Investigate failed benchmark cases and rerun the suite.
- `external_provider_runtime` (medium): External-enhanced mode is degraded because one or more enabled providers do not emit real runtime signals. Remediation: Install/configure real runtime providers or run native mode until external runtimes are available.
- `calibration_gating` (medium): Production gating is disabled because labeled validation and confidence intervals are insufficient. Remediation: Provide labeled validation samples and confidence intervals; keep recommended_for_gating false until then.

## Metrics
- `false_clean_count`: 0
- `false_incomplete_count`: 0
- `advisory_primary_failure_count`: 0
- `retrieval_security_drift_count`: 0
- `non_clean_missing_fix_count`: 0
- `failed_case_missing_first_failing_node_count`: 0
- `benchmark_accuracy`: 0.26666666666666666
- `benchmark_accuracy_threshold`: 0.95
- `provider_safe_to_run_external_enhanced`: False
- `calibration_statuses`: {'DETERMINISTIC': 5, 'PROVISIONAL': 4, 'NOT_CALIBRATED': 6}
- `full_pytest_status`: failed
- `decision_policy_status`: passed
- `external_alignment_status`: passed
- `common_benchmark_pass_rate`: 0.8913043478260869
- `subtle_benchmark_status`: failed
- `claim_harness_status`: passed
- `false_security_count`: 0
- `external_ignored_count`: 0
- `missing_provider_reason_missing_count`: 0
- `calibrated_confidence_present_count`: 0
- `production_gating_eligible`: False
- `recommended_for_gating_true_count`: 0
- `v0_1_alpha_clean_ready`: True
- `production_readiness_status`: Not Ready

## Warnings
- deepeval: No runner or mock results configured. Native runtime execution not implemented.
- ragas: No runner or mock results configured. Native runtime execution not implemented.
- refchecker_claim: RefChecker package/module could not be imported.
- refchecker_citation: RefChecker package/module could not be imported.
- ragchecker: RAGChecker package/module could not be imported.
