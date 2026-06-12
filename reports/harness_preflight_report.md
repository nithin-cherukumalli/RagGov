# Harness Preflight Report

- Status: `warn`
- Branch: `main`
- Last commit: `0f1367a`
- Recommended action: Review warnings, then proceed only with a narrow harness-safe patch.

## Dirty Files
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/cli.py`
- `tests/cli/test_diagnose_text_format.py`

## Deleted Tracked Files
- None

## Untracked Files
- `tests/cli/test_diagnose_text_format.py`

## Protected Changes
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `src/raggov/cli.py`
- `tests/cli/test_diagnose_text_format.py`

## Threshold Or Gate Changes
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/cli.py`
- `tests/cli/test_diagnose_text_format.py`

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
