# Protected Baseline Workflow

GovRAG has a protected common benchmark baseline that must be restored before launch-readiness or unrelated pytest failures are triaged.

Protected baseline:

- Native common benchmark: `41/46`
- External-enhanced common benchmark: `41/46`
- `false_clean_count = 0`
- `false_security_count = 0`
- `false_incomplete_count = 0`
- Citation: `5/5`
- Grounding: `7/7`
- Sufficiency: `5/5`
- Version validity: `5/5`
- `production_gating_eligible = false`
- Calibration status: `not_production_calibrated`

## Required Order

1. Run `python scripts/workspace_audit.py`.
2. Run `python scripts/harness_preflight.py`.
3. For baseline-sensitive work, run `python scripts/check_protected_baseline.py`.
4. Do not continue to launch-readiness blockers until the protected common baseline is recovered or the regression is explicitly explained.

## Dirty Workspace Rules

- Keep baseline-restoration source changes separate from reports, harness work, calibration work, and debug scripts.
- Do not change benchmark labels, fixtures, golden outputs, thresholds, gates, or `production_gating_eligible` without explicit user instruction.
- If reports are regenerated from a dirty workspace, state that in the final response.
- If common benchmark output regresses, stop and diagnose the implementation path before touching labels.

## Expected Remaining Common Failures

At the protected `41/46` baseline, only these five common cases should fail:

- `retrieval_top_k_too_small_08`
- `retrieval_irrelevant_plausible_09`
- `security_retrieval_anomaly_only_36`
- `quality_incomplete_38`
- `quality_ignores_context_41`

Any other common failure is a baseline regression until proven otherwise.
