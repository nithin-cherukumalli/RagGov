# Harness Preflight Report

- Status: `warn`
- Branch: `main`
- Last commit: `d74791b`
- Recommended action: Review warnings, then proceed only with a narrow harness-safe patch.

## Dirty Files
- `reports/common_failure_coverage_matrix.md`
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`
- `scripts/check_protected_baseline.py`
- `src/raggov/analyzers/citation_faithfulness/analyzer.py`
- `src/raggov/analyzers/retrieval/scope.py`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
- `src/raggov/analyzers/security/anomalies.py`
- `src/raggov/models/diagnosis.py`
- `tests/test_analyzers/test_citation_faithfulness_v0.py`
- `tests/test_analyzers/test_retrieval_profile_integration.py`
- `tests/test_analyzers/test_security.py`
- `reports/baseline_pin_v0_1_alpha_public_migration.md`
- `reports/codex_session/`
- `reports/forensics_v0_1_warn_promotion_pre_registration.md`
- `reports/forensics_v0_1_warn_promotion_result.md`
- `tests/test_analyzers/test_analyzer_calibration.py`

## Deleted Tracked Files
- None

## Untracked Files
- `reports/baseline_pin_v0_1_alpha_public_migration.md`
- `reports/codex_session/`
- `reports/forensics_v0_1_warn_promotion_pre_registration.md`
- `reports/forensics_v0_1_warn_promotion_result.md`
- `tests/test_analyzers/test_analyzer_calibration.py`

## Protected Changes
- `reports/baseline_pin_v0_1_alpha_public_migration.md`
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/forensics_v0_1_warn_promotion_pre_registration.md`
- `reports/forensics_v0_1_warn_promotion_result.md`
- `scripts/check_protected_baseline.py`

## Threshold Or Gate Changes
- `reports/baseline_pin_v0_1_alpha_public_migration.md`
- `reports/forensics_v0_1_warn_promotion_pre_registration.md`
- `reports/forensics_v0_1_warn_promotion_result.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/analyzers/retrieval/scope.py`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
- `src/raggov/analyzers/security/anomalies.py`
- `src/raggov/models/diagnosis.py`
- `tests/test_analyzers/test_security.py`

## Benchmark Summary
```json
{
  "run_common": false
}
```

## Recent Reports
```json
{
  "common_failure_triage_json": true,
  "common_failure_triage_md": true,
  "launch_readiness_json": true,
  "launch_readiness_md": true
}
```

## Commands Run
- None

## Warnings
- `Workspace has dirty files before edits.`
- `Protected benchmark, fixture, golden, or report paths are changed.`
- `Threshold, launch-readiness, or production-gating related files are changed.`

## Errors
- None
