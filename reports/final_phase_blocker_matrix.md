# Final Phase Blocker Matrix

## Target Decision

Chosen target: **v0.1-alpha-clean**

Reason: common benchmark is stable at `41/46` in native and external-enhanced modes with false clean/security/incomplete counts all `0`. Full pytest and launch readiness are not clean, and calibration is incomplete, so v0.1-rc and production-calibrated are not honest current targets.

Exact next action: **FINALIZE_REPORTS_CLEANUP_FIRST**

## Blockers

| ID | Category | Severity | Blocks v0.1 | Evidence | Smallest safe next action |
| --- | --- | --- | --- | --- | --- |
| B001 | reports_cleanup_state | medium | yes | 11 untracked generated reports; protected report outputs include `common_failure_triage.*` and `launch_readiness_report.*`. | Decide current root reports versus archive candidates; no source edits. |
| B002 | harness_failure | medium | yes | `workspace_audit`, `harness_preflight`, and `harness_post_edit_validation` all returned `warn`. | Classify generated reports and rerun harness from clean report state. |
| B003 | launch_readiness_blocker | high | production only | `launch_readiness.py` exited 1 with Status: Not Ready. | Keep production gating false and document blockers. |
| B004 | full_pytest_failure | high | production only | Full pytest failed: `50 failed, 1616 passed, 3 skipped, 31 warnings`. | Triage failures by source regression versus stale/experimental expectation. |
| B005 | subtle_suite_failure | high | production only | Launch readiness reports subtle suite failed at `20%`. | Treat subtle suite as non-gating/advisory for v0.1-alpha-clean unless targeting rc. |
| B006 | external_provider_degraded | medium | production only | `deepeval` and `ragas` degraded; RefChecker/RAGChecker unavailable. | Document degraded advisory mode. |
| B007 | calibration_incomplete | medium | production only | No calibration artifact, no labeled samples, no confidence intervals. | Document not production calibrated; keep production gating false. |
| B008 | common_benchmark_regression | low | no | No regression: native and external-enhanced are both `41/46`; false counts are zero. | Protect result; do not change labels or thresholds. |
| B009 | deleted_tracked_files | low | no | `git ls-files --deleted` returned empty. | No restore action required. |
| B010 | scratch_files | low | no | No scratch files in diagnostic dirty state. | No scratch deletion required. |

## Classification Notes

- There are no deleted tracked files.
- There are no dirty source files.
- There are no dirty test files.
- There are no observed fixture, golden label, threshold, launch gate, or production-gating changes.
- Harness warnings are caused by generated report outputs and protected report path policy, not by analyzer or engine edits.
