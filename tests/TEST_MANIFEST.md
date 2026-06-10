# Test Manifest

Generated: `2026-06-03`

This manifest documents test ownership for cleanup only. It does not change test behavior, expected labels, thresholds, gates, skips, or xfails.

## Suites
- `tests/harness/`: Harness guardrail tests for workspace/preflight/post-edit scripts.
- `tests/stresslab/`: Core GovRAG behavioral, launch-readiness, false-clean, external advisory, and pinpoint coverage. Collect with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in this sandbox.
- `tests/test_analyzers/` and `tests/analyzers/`: Analyzer unit and regression tests.
- `tests/decision_policy/`: Primary failure selection and signal-strength policy guards.
- `tests/engine/` plus root engine tests: Diagnosis engine orchestration and structured output tests.
- `tests/evals/` and `tests/evaluation/`: Calibration/evaluation harness tests; review before changing baselines or readiness semantics.
- `tests/evaluators/` and `tests/external_alignment/`: Optional external provider adapter/alignment tests; external providers are advisory.
- `tests/fixtures/` and `tests/Data/`: Fixture/golden-style test data; protected from cleanup edits.

## Cleanup Rule
Generated caches may be cleaned in a separate execution pass if approved. Test files and fixtures require case-level review before deletion or semantic edits.
