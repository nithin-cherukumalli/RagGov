# v0.1-alpha-clean Tag Readiness

## Final Recommendation

`READY_FOR_HUMAN_COMMIT_AND_TAG`

## Scope

This is a human-review style dirty-file audit for tagging `v0.1-alpha-clean`.

No `git add`, commit, or tag command was run.

## Final Validation Results

- `python scripts/check_v0_1_alpha_release.py`: passed.
- `python scripts/evaluate_common_failures.py --suite common`: `41/46` passed.
- `python scripts/evaluate_common_failures.py --suite common --mode external-enhanced`: `41/46` passed.
- `python scripts/launch_readiness.py`: exit `0`; status `v0.1-alpha-clean Ready`.
- `pytest -q tests/harness`: `13 passed`.
- `pytest -q tests/decision_policy`: `45 passed`.
- `pytest -q tests/test_analyzers/test_grounding.py`: `51 passed`.
- `git ls-files --deleted`: no output; deleted tracked files count `0`.

## Alpha Gate Truth

- native common benchmark: `41/46`
- external-enhanced common benchmark: `41/46`
- `false_clean_count`: `0`
- `false_security_count`: `0`
- `false_incomplete_count`: `0`
- `advisory_primary_failure_count`: `0`
- `retrieval_security_drift_count`: `0`
- `production_gating_eligible`: `false`
- production readiness status: `Not Ready`
- full pytest status: `failed`; RC blocker
- subtle benchmark status: `failed`; advisory/RC-level blocker

## Dirty Summary

`git diff --stat`:

```text
10 files changed, 337 insertions(+), 14 deletions(-)
```

Tracked dirty files are alpha implementation and test changes from the prior alpha work. Untracked files are release docs/scripts/reports plus generated validation artifacts.

## Protected No-Touch Check

The following protected/no-touch areas were checked and had no dirty files:

- `fixtures/**`
- `stresslab/cases/golden/**`
- `stresslab/cases/**`
- `pyproject.toml`
- requirements files
- setup/config files checked in this audit
- `.github/**`
- `harness/protected_baseline.json`
- `harness/failure_mode_registry.json`

No fixture, golden-label, dependency, threshold/config, or production-gating file was dirty in this check.

## A. INCLUDE_IN_ALPHA_COMMIT

| Path | Status | Reason | safe_to_commit | Risk |
| --- | --- | --- | --- | --- |
| `src/raggov/analyzers/citation_faithfulness/analyzer.py` | modified | Alpha fix for citation support behavior; covered by targeted tests and alpha gate. | true | medium |
| `src/raggov/analyzers/grounding/claims.py` | modified | Alpha fix for structured claim extraction normalization; covered by grounding tests and alpha gate. | true | medium |
| `src/raggov/analyzers/grounding/support.py` | modified | Alpha fix for claim support behavior; covered by grounding tests and alpha gate. | true | medium |
| `src/raggov/analyzers/security/anomalies.py` | modified | Alpha fix for retrieval anomaly/security drift behavior; covered by security-related alpha tests and release gate. | true | medium |
| `src/raggov/evaluators/claim/structured_llm.py` | modified | Alpha fix preserving structured claim verifier metadata; covered by alpha validation. | true | medium |
| `stresslab/runners/launch_readiness.py` | modified | Alpha launch-readiness classification of alpha/RC/production blockers. | true | medium |
| `tests/stresslab/test_launch_readiness.py` | modified | Tests for alpha launch-readiness classification. | true | low |
| `tests/test_analyzers/test_citation_faithfulness_v0.py` | modified | Tests for citation support alpha fixes. | true | low |
| `tests/test_analyzers/test_grounding.py` | modified | Tests for grounding and claim extraction alpha fixes. | true | low |
| `tests/test_analyzers/test_security.py` | modified | Tests for retrieval anomaly/security drift alpha fixes. | true | low |
| `docs/V0_1_ALPHA_RELEASE.md` | untracked | Alpha release documentation. | true | low |
| `docs/ROADMAP_AFTER_ALPHA.md` | untracked | Post-alpha roadmap and blocker documentation. | true | low |
| `RELEASE_NOTES_v0.1-alpha.md` | untracked | Final release notes for `v0.1-alpha-clean`. | true | low |
| `scripts/check_v0_1_alpha_release.py` | untracked | Minimal alpha release gate; final validation passed. | true | low |
| `reports/v0_1_alpha_finish_result.md` | untracked | Alpha finish evidence report from prior freeze work. | true | low |
| `reports/v0_1_alpha_freeze_final.md` | untracked | Final freeze report with `SAFE_TO_TAG_V0_1_ALPHA`. | true | low |
| `reports/v0_1_alpha_release_freeze_result.md` | untracked | Release-freeze result report. | true | low |
| `reports/v0_1_alpha_tag_readiness.md` | untracked | This tag-readiness audit report. | true | low |

## B. EXCLUDE_FROM_ALPHA_COMMIT

| Path | Status | Reason | safe_to_commit | Risk |
| --- | --- | --- | --- | --- |
| `reports/common_failure_coverage_matrix.md` | untracked | Regenerated benchmark output; reproducible from validation commands. | false | low |
| `reports/common_failure_triage.json` | untracked | Regenerated benchmark triage output; useful locally but not required in the alpha commit. | false | low |
| `reports/common_failure_triage.md` | untracked | Regenerated benchmark triage output; useful locally but not required in the alpha commit. | false | low |
| `reports/launch_readiness_report.json` | untracked | Regenerated launch-readiness output; reproducible from `python scripts/launch_readiness.py`. | false | low |
| `reports/launch_readiness_report.md` | untracked | Regenerated launch-readiness output; reproducible from `python scripts/launch_readiness.py`. | false | low |
| `reports/harness_preflight_report.json` | untracked | Generated harness audit artifact. | false | low |
| `reports/harness_preflight_report.md` | untracked | Generated harness audit artifact. | false | low |
| `reports/harness_post_edit_validation.json` | untracked | Generated harness audit artifact. | false | low |
| `reports/harness_post_edit_validation.md` | untracked | Generated harness audit artifact. | false | low |
| `reports/workspace_audit.json` | untracked | Generated workspace audit artifact. | false | low |
| `reports/workspace_audit.md` | untracked | Generated workspace audit artifact. | false | low |
| `reports/final_phase_blocker_matrix.json` | untracked | Earlier generated planning/audit artifact; not required for final tag commit. | false | low |
| `reports/final_phase_blocker_matrix.md` | untracked | Earlier generated planning/audit artifact; not required for final tag commit. | false | low |
| `reports/v0_1_alpha_finish_plan.md` | untracked | Earlier planning artifact; not required for final tag commit. | false | low |

## C. HUMAN_REVIEW_BEFORE_COMMIT

No dirty protected files, dependency files, fixture files, golden files, threshold/config files, or production-gating files were found.

The only judgment call is whether the team wants to commit regenerated benchmark/launch-readiness report artifacts. This audit excludes them from the suggested commit because they are generated and reproducible.

## Suggested Commands

Suggested `git add` command for `INCLUDE_IN_ALPHA_COMMIT` files only:

```bash
git add \
  src/raggov/analyzers/citation_faithfulness/analyzer.py \
  src/raggov/analyzers/grounding/claims.py \
  src/raggov/analyzers/grounding/support.py \
  src/raggov/analyzers/security/anomalies.py \
  src/raggov/evaluators/claim/structured_llm.py \
  stresslab/runners/launch_readiness.py \
  tests/stresslab/test_launch_readiness.py \
  tests/test_analyzers/test_citation_faithfulness_v0.py \
  tests/test_analyzers/test_grounding.py \
  tests/test_analyzers/test_security.py \
  docs/V0_1_ALPHA_RELEASE.md \
  docs/ROADMAP_AFTER_ALPHA.md \
  RELEASE_NOTES_v0.1-alpha.md \
  scripts/check_v0_1_alpha_release.py \
  reports/v0_1_alpha_finish_result.md \
  reports/v0_1_alpha_freeze_final.md \
  reports/v0_1_alpha_release_freeze_result.md \
  reports/v0_1_alpha_tag_readiness.md
```

Suggested commit command:

```bash
git commit -m "Prepare GovRAG v0.1-alpha-clean release"
```

Suggested tag command:

```bash
git tag -a v0.1-alpha-clean -m "GovRAG v0.1-alpha-clean"
```

Suggested verification command:

```bash
git show --stat v0.1-alpha-clean
```

## Tag Decision

Safe to tag after human runs the suggested add/commit/tag commands: yes.

Do not tag as production-ready, production-calibrated, or production-gating eligible.
