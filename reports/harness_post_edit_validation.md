# Harness Post-Edit Validation

- Status: `warn`
- Recommended action: Stop and investigate baseline regressions or protected edits.

## Changed Files
- `evals/govrag_calib/splits/README.md`
- `scripts/evaluate_govrag_calib.py`
- `scripts/validate_govrag_calib.py`
- `src/raggov/analyzers/grounding/claims.py`
- `src/raggov/analyzers/security/privacy.py`
- `src/raggov/models/diagnosis.py`
- `tests/evals/test_govrag_calib_schema.py`
- `tests/test_analyzers/test_grounding.py`
- `tests/test_analyzers/test_security.py`
- `evals/govrag_calib/ADJUDICATION_GUIDE.md`
- `evals/govrag_calib/CASE_AUTHORING_GUIDE.md`
- `evals/govrag_calib/EXPANSION_PLAN.md`
- `evals/govrag_calib/README.md`
- `evals/govrag_calib/calib_150_seed.jsonl`
- `evals/govrag_calib/results/`
- `evals/govrag_calib/schema.json`
- `evals/govrag_calib/templates/`
- `evals/govrag_calib/validation/`
- `reports/common_failure_coverage_matrix.md`
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/final_phase_blocker_matrix.json`
- `reports/final_phase_blocker_matrix.md`
- `reports/govrag_calib_50_adjudication_diagnostics_result.md`
- `reports/govrag_calib_50_error_triage.json`
- `reports/govrag_calib_50_error_triage.md`
- `reports/govrag_calib_50_expansion_result.md`
- `reports/govrag_calib_50_safety_repair_result.md`
- `reports/govrag_calib_50_security_scoring_semantics_result.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/launch_readiness_report.json`
- `reports/launch_readiness_report.md`
- `reports/v0_1_alpha_finish_plan.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`
- `scripts/summarize_govrag_calib.py`
- `tests/evals/test_govrag_calib_evaluator.py`
- `tests/evals/test_govrag_calib_summary.py`

## Risk Classification
```json
{
  "critical": [
    "evals/govrag_calib/ADJUDICATION_GUIDE.md",
    "evals/govrag_calib/CASE_AUTHORING_GUIDE.md",
    "evals/govrag_calib/EXPANSION_PLAN.md",
    "evals/govrag_calib/README.md",
    "evals/govrag_calib/calib_150_seed.jsonl",
    "evals/govrag_calib/results/",
    "evals/govrag_calib/schema.json",
    "evals/govrag_calib/splits/README.md",
    "evals/govrag_calib/templates/",
    "evals/govrag_calib/validation/",
    "reports/common_failure_triage.json",
    "reports/common_failure_triage.md",
    "reports/launch_readiness_report.json",
    "reports/launch_readiness_report.md"
  ],
  "high": [
    "src/raggov/analyzers/grounding/claims.py",
    "src/raggov/analyzers/security/privacy.py"
  ],
  "low": [
    "reports/common_failure_coverage_matrix.md",
    "reports/final_phase_blocker_matrix.json",
    "reports/final_phase_blocker_matrix.md",
    "reports/govrag_calib_50_adjudication_diagnostics_result.md",
    "reports/govrag_calib_50_error_triage.json",
    "reports/govrag_calib_50_error_triage.md",
    "reports/govrag_calib_50_expansion_result.md",
    "reports/govrag_calib_50_safety_repair_result.md",
    "reports/govrag_calib_50_security_scoring_semantics_result.md",
    "reports/harness_post_edit_validation.json",
    "reports/harness_post_edit_validation.md",
    "reports/harness_preflight_report.json",
    "reports/harness_preflight_report.md",
    "reports/v0_1_alpha_finish_plan.md",
    "reports/workspace_audit.json",
    "reports/workspace_audit.md",
    "tests/evals/test_govrag_calib_evaluator.py",
    "tests/evals/test_govrag_calib_schema.py",
    "tests/evals/test_govrag_calib_summary.py",
    "tests/test_analyzers/test_grounding.py",
    "tests/test_analyzers/test_security.py"
  ],
  "medium": [
    "scripts/evaluate_govrag_calib.py",
    "scripts/summarize_govrag_calib.py",
    "scripts/validate_govrag_calib.py",
    "src/raggov/models/diagnosis.py"
  ]
}
```

## Commands Run
- None

## Benchmark Before Or Baseline
```json
{
  "calibration_status": "not_production_calibrated",
  "citation": "5/5",
  "common_external_passed": 41,
  "common_external_total": 46,
  "common_native_passed": 41,
  "common_native_total": 46,
  "description": "Harness reference baseline only. Production logic must not depend on this file.",
  "false_clean_count": 0,
  "false_incomplete_count": 0,
  "false_security_count": 0,
  "grounding": "7/7",
  "production_gating_eligible": false,
  "schema_version": 1,
  "sufficiency": "5/5",
  "version_validity": "5/5"
}
```

## Benchmark After
```json
{
  "generated_at": "2026-06-11T06:59:47.929279+00:00",
  "modes": {
    "external-enhanced": {
      "category_stats": {
        "answer_quality": {
          "pass_rate": 0.6666666666666666,
          "passed": 4,
          "total": 6
        },
        "citation": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        },
        "grounding": {
          "pass_rate": 0.8571428571428571,
          "passed": 6,
          "total": 7
        },
        "parser_chunking": {
          "pass_rate": 1.0,
          "passed": 6,
          "total": 6
        },
        "retrieval": {
          "pass_rate": 0.8333333333333334,
          "passed": 5,
          "total": 6
        },
        "security": {
          "pass_rate": 0.8333333333333334,
          "passed": 5,
          "total": 6
        },
        "sufficiency": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        },
        "version_validity": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        }
      },
      "failures": [
        {
          "actual_first_failing_node": "retrieval_coverage",
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "retrieval_top_k_too_small_08",
          "category": "retrieval",
          "expected_first_failing_node": null,
          "expected_fix": "INCREASE_TOP_K",
          "expected_primary_failure": "RETRIEVAL_DEPTH_LIMIT",
          "expected_stage": "RETRIEVAL",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Retrieval evidence is generated but subtype mapping remains too coarse for this fixture.",
          "likely_failing_analyzer": "RetrievalDiagnosisAnalyzerV0"
        },
        {
          "actual_first_failing_node": "retrieval_coverage",
          "actual_fix": "Expand retrieval to include effective dates, version metadata, or current lifecycle state before answering.",
          "actual_primary_failure": "INSUFFICIENT_CONTEXT",
          "actual_stage": "SUFFICIENCY",
          "case_id": "grounding_date_hallucination_20",
          "category": "grounding",
          "expected_first_failing_node": null,
          "expected_fix": "TEMPORAL_VERIFICATION",
          "expected_primary_failure": "UNSUPPORTED_CLAIM",
          "expected_stage": "GROUNDING",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer (emitted expected evidence)"
        },
        {
          "actual_first_failing_node": "context_assembly",
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "security_retrieval_anomaly_only_36",
          "category": "security",
          "expected_first_failing_node": null,
          "expected_fix": "RETRIEVAL_DIAGNOSIS",
          "expected_primary_failure": "RETRIEVAL_ANOMALY",
          "expected_stage": "RETRIEVAL",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "RetrievalAnomalyAnalyzer (warn-level evidence not final)"
        },
        {
          "actual_first_failing_node": "claim_support",
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "quality_incomplete_38",
          "category": "answer_quality",
          "expected_first_failing_node": null,
          "expected_fix": "COMPLETENESS_VERIFICATION",
          "expected_primary_failure": "UNSUPPORTED_CLAIM",
          "expected_stage": "GENERATION",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0"
        },
        {
          "actual_first_failing_node": "retrieval_coverage",
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "quality_ignores_context_41",
          "category": "answer_quality",
          "expected_first_failing_node": null,
          "expected_fix": "GROUNDING_PROMPT_FIX",
          "expected_primary_failure": "CONTRADICTED_CLAIM",
          "expected_stage": "GENERATION",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0"
        }
      ],
      "false_clean_count": 0,
      "false_incomplete_count": 0,
      "false_security_count": 0,
      "mode": "external-enhanced",
      "pass_rate": 0.8913043478260869,
      "passed_cases": 41,
      "total_cases": 46
    },
    "native": {
      "category_stats": {
        "answer_quality": {
          "pass_rate": 0.6666666666666666,
          "passed": 4,
          "total": 6
        },
        "citation": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        },
        "grounding": {
          "pass_rate": 0.8571428571428571,
          "passed": 6,
          "total": 7
        },
        "parser_chunking": {
          "pass_rate": 1.0,
          "passed": 6,
          "total": 6
        },
        "retrieval": {
          "pass_rate": 0.8333333333333334,
          "passed": 5,
          "total": 6
        },
        "security": {
          "pass_rate": 0.8333333333333334,
          "passed": 5,
          "total": 6
        },
        "sufficiency": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        },
        "version_validity": {
          "pass_rate": 1.0,
          "passed": 5,
          "total": 5
        }
      },
      "failures": [
        {
          "actual_first_failing_node": null,
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "retrieval_top_k_too_small_08",
          "category": "retrieval",
          "expected_first_failing_node": null,
          "expected_fix": "INCREASE_TOP_K",
          "expected_primary_failure": "RETRIEVAL_DEPTH_LIMIT",
          "expected_stage": "RETRIEVAL",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Retrieval evidence is generated but subtype mapping remains too coarse for this fixture.",
          "likely_failing_analyzer": "RetrievalDiagnosisAnalyzerV0"
        },
        {
          "actual_first_failing_node": null,
          "actual_fix": "Expand retrieval to include effective dates, version metadata, or current lifecycle state before answering.",
          "actual_primary_failure": "INSUFFICIENT_CONTEXT",
          "actual_stage": "SUFFICIENCY",
          "case_id": "grounding_date_hallucination_20",
          "category": "grounding",
          "expected_first_failing_node": null,
          "expected_fix": "TEMPORAL_VERIFICATION",
          "expected_primary_failure": "UNSUPPORTED_CLAIM",
          "expected_stage": "GROUNDING",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer (emitted expected evidence)"
        },
        {
          "actual_first_failing_node": null,
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "security_retrieval_anomaly_only_36",
          "category": "security",
          "expected_first_failing_node": null,
          "expected_fix": "RETRIEVAL_DIAGNOSIS",
          "expected_primary_failure": "RETRIEVAL_ANOMALY",
          "expected_stage": "RETRIEVAL",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "RetrievalAnomalyAnalyzer (warn-level evidence not final)"
        },
        {
          "actual_first_failing_node": null,
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "quality_incomplete_38",
          "category": "answer_quality",
          "expected_first_failing_node": null,
          "expected_fix": "COMPLETENESS_VERIFICATION",
          "expected_primary_failure": "UNSUPPORTED_CLAIM",
          "expected_stage": "GENERATION",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0"
        },
        {
          "actual_first_failing_node": null,
          "actual_fix": "1 of 1 claims are unsupported by retrieved context. Review retrieval quality or add source verification.",
          "actual_primary_failure": "UNSUPPORTED_CLAIM",
          "actual_stage": "GROUNDING",
          "case_id": "quality_ignores_context_41",
          "category": "answer_quality",
          "expected_first_failing_node": null,
          "expected_fix": "GROUNDING_PROMPT_FIX",
          "expected_primary_failure": "CONTRADICTED_CLAIM",
          "expected_stage": "GENERATION",
          "external_probes": [],
          "false_clean": false,
          "false_incomplete": false,
          "false_security": false,
          "likely_code_cause": "Expected evidence exists but decision policy selected a different higher-ranked failure.",
          "likely_failing_analyzer": "ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0"
        }
      ],
      "false_clean_count": 0,
      "false_incomplete_count": 0,
      "false_security_count": 0,
      "mode": "native",
      "pass_rate": 0.8913043478260869,
      "passed_cases": 41,
      "total_cases": 46
    }
  },
  "suite": "common"
}
```

## Mode Results
```json
{
  "external_enhanced": {
    "false_clean_count": 0,
    "false_incomplete_count": 0,
    "false_security_count": 0,
    "passed": 41,
    "total": 46
  },
  "native": {
    "false_clean_count": 0,
    "false_incomplete_count": 0,
    "false_security_count": 0,
    "passed": 41,
    "total": 46
  }
}
```

## False Clean Count
`0`

## False Security Count
`0`

## False Incomplete Count
`0`

## Production Gating Eligible
`False`

## Protected Changes
- `evals/govrag_calib/ADJUDICATION_GUIDE.md`
- `evals/govrag_calib/CASE_AUTHORING_GUIDE.md`
- `evals/govrag_calib/EXPANSION_PLAN.md`
- `evals/govrag_calib/README.md`
- `evals/govrag_calib/calib_150_seed.jsonl`
- `evals/govrag_calib/results/`
- `evals/govrag_calib/schema.json`
- `evals/govrag_calib/splits/README.md`
- `evals/govrag_calib/templates/`
- `evals/govrag_calib/validation/`
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/final_phase_blocker_matrix.json`
- `reports/final_phase_blocker_matrix.md`
- `reports/govrag_calib_50_adjudication_diagnostics_result.md`
- `reports/govrag_calib_50_error_triage.json`
- `reports/govrag_calib_50_error_triage.md`
- `reports/govrag_calib_50_expansion_result.md`
- `reports/govrag_calib_50_safety_repair_result.md`
- `reports/govrag_calib_50_security_scoring_semantics_result.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/launch_readiness_report.json`
- `reports/launch_readiness_report.md`
- `reports/v0_1_alpha_finish_plan.md`
- `scripts/evaluate_govrag_calib.py`
- `scripts/summarize_govrag_calib.py`
- `scripts/validate_govrag_calib.py`
- `tests/evals/test_govrag_calib_evaluator.py`
- `tests/evals/test_govrag_calib_schema.py`
- `tests/evals/test_govrag_calib_summary.py`

## Threshold Or Gate Changes
- `evals/govrag_calib/CASE_AUTHORING_GUIDE.md`
- `evals/govrag_calib/README.md`
- `evals/govrag_calib/calib_150_seed.jsonl`
- `evals/govrag_calib/splits/README.md`
- `reports/final_phase_blocker_matrix.json`
- `reports/final_phase_blocker_matrix.md`
- `reports/govrag_calib_50_adjudication_diagnostics_result.md`
- `reports/govrag_calib_50_error_triage.json`
- `reports/govrag_calib_50_error_triage.md`
- `reports/govrag_calib_50_expansion_result.md`
- `reports/govrag_calib_50_safety_repair_result.md`
- `reports/govrag_calib_50_security_scoring_semantics_result.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/launch_readiness_report.json`
- `reports/launch_readiness_report.md`
- `reports/v0_1_alpha_finish_plan.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`
- `scripts/evaluate_govrag_calib.py`
- `scripts/summarize_govrag_calib.py`
- `scripts/validate_govrag_calib.py`
- `src/raggov/analyzers/grounding/claims.py`
- `src/raggov/models/diagnosis.py`
- `tests/evals/test_govrag_calib_evaluator.py`
- `tests/evals/test_govrag_calib_schema.py`
- `tests/evals/test_govrag_calib_summary.py`
- `tests/test_analyzers/test_grounding.py`
- `tests/test_analyzers/test_security.py`

## Protected Baseline Regressions
- None

## Warnings
- `Protected benchmark, fixture, golden, or report paths are changed.`
- `Threshold, launch-readiness, or production-gating related files are changed.`
- `Critical-risk files changed.`

## Errors
- None
