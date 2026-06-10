# GovRAG Agent Workspace Harness

The harness gives coding agents a small, repeatable safety check around GovRAG development. It is audit-only: it reports workspace risk, protected benchmark changes, threshold or gate edits, and baseline regressions without changing product logic.

## When To Run

- Run preflight before edits: `python scripts/harness_preflight.py`
- Run post-edit validation after edits: `python scripts/harness_post_edit_validation.py`
- Run a standalone workspace check when taking over a dirty tree: `python scripts/workspace_audit.py`

Optional benchmark commands are behind flags because they can be slower:

```bash
python scripts/harness_preflight.py --run-common
python scripts/harness_post_edit_validation.py --run-common --run-launch
```

## Reports

Each script writes JSON and Markdown under `reports/`:

- `reports/harness_preflight_report.json`
- `reports/harness_preflight_report.md`
- `reports/harness_post_edit_validation.json`
- `reports/harness_post_edit_validation.md`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`

JSON is for automation. Markdown is for human review. Missing optional reports are warnings unless `--strict` is used.

## Agent Workflow

1. Run `python scripts/workspace_audit.py` to see dirty state.
2. Run `python scripts/harness_preflight.py` before editing.
3. Make a narrow change.
4. Run targeted tests for the changed component.
5. Run `python scripts/harness_post_edit_validation.py`.
6. Report files changed, commands run, protected-file status, gating status, and limitations.

## Safe PR Examples

- Add harness scripts, docs, and tests.
- Add explicit metadata to an existing analyzer with targeted tests.
- Generate additive audit reports that preserve failure details.

## Unsafe PR Examples

- Change `expected_primary_failure`, `expected_stage`, `expected_root_cause`, or `expected_fix_category` without explicit approval.
- Edit thresholds or launch-readiness gates to improve reported status.
- Enable `production_gating_eligible`.
- Mark heuristic or proxy signals as calibrated.
- Rewrite reports to hide degraded external runtime or known benchmark blockers.

## Minimal Command Sequence

```bash
python scripts/workspace_audit.py
python scripts/harness_preflight.py
pytest -q tests/harness
python scripts/harness_post_edit_validation.py
```
