# Harness Preflight Report

- Status: `warn`
- Branch: `main`
- Last commit: `64909ff`
- Recommended action: Review warnings, then proceed only with a narrow harness-safe patch.

## Dirty Files
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/analyzers/retrieval/stale.py`
- `reports/forensics_v0_1_stale_retrieval_pre_registration.md`
- `tests/test_analyzers/test_stale_retrieval_relative_recency.py`

## Deleted Tracked Files
- None

## Untracked Files
- `reports/forensics_v0_1_stale_retrieval_pre_registration.md`
- `tests/test_analyzers/test_stale_retrieval_relative_recency.py`

## Protected Changes
- `reports/common_failure_triage.json`
- `reports/common_failure_triage.md`
- `reports/forensics_v0_1_stale_retrieval_pre_registration.md`

## Threshold Or Gate Changes
- `reports/forensics_v0_1_stale_retrieval_pre_registration.md`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `src/raggov/analyzers/retrieval/stale.py`
- `tests/test_analyzers/test_stale_retrieval_relative_recency.py`

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
