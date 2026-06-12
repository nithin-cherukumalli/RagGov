# Harness Preflight Report

- Status: `warn`
- Branch: `main`
- Last commit: `cf0fc4e`
- Recommended action: Review warnings, then proceed only with a narrow harness-safe patch.

## Dirty Files
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`
- `src/raggov/cli.py`
- `src/raggov/models/diagnosis.py`
- `evals/govrag_calib/splits/heldout_v0_1.json`
- `evals/govrag_calib/splits/heldout_v0_1.jsonl`
- `reports/baseline_pin_v0_1_alpha_public_decision.md`
- `reports/calib50_step2_result.json`
- `reports/calib50_step2_result.md`
- `tests/test_cli/`
- `tests/test_models/test_human_review_escalation.py`

## Deleted Tracked Files
- None

## Untracked Files
- `evals/govrag_calib/splits/heldout_v0_1.json`
- `evals/govrag_calib/splits/heldout_v0_1.jsonl`
- `reports/baseline_pin_v0_1_alpha_public_decision.md`
- `reports/calib50_step2_result.json`
- `reports/calib50_step2_result.md`
- `tests/test_cli/`
- `tests/test_models/test_human_review_escalation.py`

## Protected Changes
- `evals/govrag_calib/splits/heldout_v0_1.json`
- `evals/govrag_calib/splits/heldout_v0_1.jsonl`
- `reports/baseline_pin_v0_1_alpha_public_decision.md`
- `reports/calib50_step2_result.json`
- `reports/calib50_step2_result.md`
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `src/raggov/cli.py`

## Threshold Or Gate Changes
- `evals/govrag_calib/splits/heldout_v0_1.jsonl`
- `reports/baseline_pin_v0_1_alpha_public_decision.md`
- `reports/calib50_step2_result.json`
- `reports/calib50_step2_result.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/cli.py`
- `src/raggov/models/diagnosis.py`

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
