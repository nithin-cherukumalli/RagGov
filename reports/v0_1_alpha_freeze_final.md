# v0.1-alpha-clean Final Freeze Report

## Final Recommendation

`SAFE_TO_TAG_V0_1_ALPHA`

## Decision

GovRAG is safe to tag as `v0.1-alpha-clean` after final human review of the dirty-file scope.

This is an alpha freeze only. Production remains `Not Ready`, `production_gating_eligible` remains `false`, and the release is not production-calibrated.

## Files Changed

- `RELEASE_NOTES_v0.1-alpha.md`: added final alpha release notes.
- `reports/v0_1_alpha_freeze_final.md`: added this final freeze report.

Existing release docs and release gate were checked and left unchanged:

- `docs/V0_1_ALPHA_RELEASE.md`
- `docs/ROADMAP_AFTER_ALPHA.md`
- `scripts/check_v0_1_alpha_release.py`

No source, fixture, golden-label, threshold, decision-policy, engine, analyzer, or production-gating changes were made for this freeze step.

## Commands Run

- `python scripts/evaluate_common_failures.py --suite common`: `41/46` passed.
- `python scripts/evaluate_common_failures.py --suite common --mode external-enhanced`: `41/46` passed.
- `python scripts/launch_readiness.py`: exit `0`; status `v0.1-alpha-clean Ready`.
- `python scripts/check_v0_1_alpha_release.py || true`: alpha release gate passed.
- `pytest -q tests/harness`: `13 passed`.
- `pytest -q tests/decision_policy`: `45 passed`.
- `pytest -q tests/test_analyzers/test_grounding.py`: `51 passed`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --collect-only -q tests/stresslab`: `270 tests collected`; known integration mark warnings.
- `git status --short`: dirty tracked alpha implementation/test files remain, plus generated reports and release artifacts.
- `git ls-files --deleted`: no deleted tracked files.

## Validation Results

- native common benchmark: `41/46`
- external-enhanced common benchmark: `41/46`
- `false_clean_count`: `0`
- `false_security_count`: `0`
- `false_incomplete_count`: `0`
- `advisory_primary_failure_count`: `0`
- `retrieval_security_drift_count`: `0`
- `production_gating_eligible`: `false`
- launch readiness status: `v0.1-alpha-clean Ready`
- launch readiness exit status: `0`
- production readiness status: `Not Ready`
- release gate status: passed
- harness status: passed
- deleted tracked files count: `0`

## Release Docs Status

- `docs/V0_1_ALPHA_RELEASE.md`: exists and states alpha is not production-calibrated, production gating remains disabled, protected benchmark status is `41/46`, false counts are `0/0/0`, external providers are advisory/degraded when unavailable, and full pytest/subtle suite are RC blockers.
- `docs/ROADMAP_AFTER_ALPHA.md`: exists and lists RC, production-calibrated, calibration, provider stabilization, harness hardening, and cleanup tasks.
- `RELEASE_NOTES_v0.1-alpha.md`: exists and includes the exact statement: “This release is not production-calibrated.”

## Release Gate Status

`scripts/check_v0_1_alpha_release.py` exists and passed.

It checks:

- native common benchmark does not drop below `41/46`
- external-enhanced common benchmark does not drop below `41/46`
- false-clean, false-security, and false-incomplete counts remain zero
- `production_gating_eligible` remains `false`
- launch readiness reports `v0.1-alpha-clean Ready`

It does not require full pytest, production calibration, external provider availability, or a `46/46` common benchmark.

## Dirty Files Summary

Tracked dirty files are prior alpha implementation/test changes, not new changes from this freeze step:

- analyzer and evaluator source files under `src/raggov/...`
- `stresslab/runners/launch_readiness.py`
- related analyzer/security/launch-readiness tests

Untracked generated report artifacts remain present. No report cleanup, archive, or deletion was performed.

## Remaining RC Blockers

- Full pytest still fails.
- Common benchmark is not `46/46`.
- Subtle suite remains advisory/RC-level.
- Answer-quality and pinpointing mismatches remain.
- Test warning and mark registration cleanup remains.

## Remaining Production Blockers

- Calibration dataset is incomplete.
- Confidence intervals and calibration artifacts are absent.
- `production_gating_eligible` must remain `false`.
- External providers remain degraded/advisory.
- Production-gating policy is not validated.
- Security and privacy review remains.

## Safe To Tag

Safe to tag: yes, for `v0.1-alpha-clean`.

Do not tag this as production-ready, production-calibrated, or production-gating eligible.
