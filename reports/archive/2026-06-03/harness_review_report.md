# Harness Review Report

- Status: `fail`
- Final recommendation: `DO_NOT_ACCEPT_PROTECTED_GUARDRAILS_WEAK`
- Safe to keep: `True`
- Should block future coding-agent PRs: `False`

## Findings
- **HIGH** [scripts/harness_post_edit_validation.py:48] Common benchmark pass totals are not compared against the protected baseline. The baseline has common_native_passed/total and common_external_passed/total, but _baseline_regressions only checks false_clean_count, false_security_count, false_incomplete_count, and production_gating_eligible. A drop from 41/46 to a lower pass count can be missed if false counts stay at zero.
- **HIGH** [scripts/harness_common.py:175] Native and external-enhanced mode results are not validated separately. find_key returns the first matching false-count key found in a nested report. For reports with modes.native and modes.external-enhanced, the selected count depends on JSON traversal order. Post-edit validation reports only one aggregate false_clean/security/incomplete value and does not verify both modes or either mode's pass total.
- **MEDIUM** [harness/protected_paths.json:15] Threshold/gate detection is useful but noisy and incomplete. It combines filename globs with content tokens, which correctly catches production_gating_eligible but also marks many generated reports and docs as threshold/gate changes. It does not explicitly protect common threshold/config locations beyond config/configs and filename matches.
- **MEDIUM** [scripts/workspace_audit.py:35] Protected changes are warnings even when safe_to_continue is false. workspace_audit sets safe_to_continue false when protected files changed, but status remains warn unless there are deleted tracked files. This is acceptable for advisory use, but too weak for a blocking harness mode.
- **MEDIUM** [harness/protected_paths.json:57] Risk classification treats all tests as low risk before critical matching can distinguish benchmark tests. The classifier checks critical first, so tests/fixtures is protected, but ordinary tests under tests/stresslab and decision-policy regression tests fall to low unless separately matched by another critical pattern. This is acceptable for general tests, but benchmark-integrity tests deserve a higher risk class.
- **LOW** [tests/harness/test_harness_scripts.py:1] Tests cover core checks but not mode-separated benchmark regression parsing. Tests are meaningful and fast, but they do not assert native/external-enhanced pass-total regression detection or degraded provider reporting because that logic is not implemented yet.

## Commands Run
- `python scripts/workspace_audit.py` -> warn
- `python scripts/harness_preflight.py` -> warn
- `python scripts/harness_post_edit_validation.py` -> warn
- `pytest -q tests/harness` -> 8 passed
- `python scripts/harness_preflight.py --run-common` -> warn
- `python scripts/harness_post_edit_validation.py --run-common` -> warn

## Files Inspected
- `AGENTS.md`
- `harness/README.md`
- `harness/failure_mode_registry.json`
- `harness/failure_mode_registry.md`
- `harness/protected_paths.json`
- `harness/protected_baseline.json`
- `scripts/harness_common.py`
- `scripts/harness_preflight.py`
- `scripts/harness_post_edit_validation.py`
- `scripts/workspace_audit.py`
- `tests/harness/test_harness_scripts.py`
- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`

## Missing Guardrails
- Compare native and external-enhanced common benchmark pass totals against protected_baseline.json.
- Report false_clean_count, false_security_count, and false_incomplete_count per mode rather than via first-key recursive lookup.
- Represent external-enhanced provider degradation from benchmark or launch reports explicitly in post-edit output.
- Add stricter blocking semantics for protected changes under --strict or for future CI usage.

## False Positives
- Content-token scanning marks many reports/docs as threshold_or_gate_changes when they merely mention production_gating_eligible or threshold.
- Generated report changes are included in protected_changes when they contain protected tokens, producing noisy preflight/post-edit output.

## False Negatives
- Common benchmark pass-count regressions are missed if false counts remain zero.
- Native-only or external-enhanced-only regressions can be missed because counts are not mode-scoped.
- External provider degradation is documented but not detected as a structured post-edit regression.

## Overengineering Concerns
- The harness is not over-engineered: it is standard-library only, small, and audit-oriented.
- The failure-mode registry is broad but still project-specific and acceptable as reference documentation.

## Recommendation
Add mode-aware common benchmark parsing in harness_post_edit_validation.py and tests that fail on native or external-enhanced pass-count regressions below 41/46.

## Integrity
- Code changed by review: `False`
- Benchmark labels changed: `False`
- Thresholds changed: `False`
- Production gating changed: `False`
