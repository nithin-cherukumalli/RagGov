# Analyzer Calibration Result

Status: `PASSED`

## Criteria

| Criterion | Result | Evidence |
|---|---|---|
| Protected common baseline: 42/46 with composition unchanged | PASSED | `python scripts/check_protected_baseline.py` exited 0. Raw run reports 41/46 plus the pinned acceptable alternative. |
| Protected false_clean_count = 0, false_security_count = 0, false_incomplete_count = 0 | PASSED | `reports/common_failure_triage.json`: native/external all 0. |
| Calib-50 false_clean_count = 0, dangerous_clean_miss_count = 0, human_review_miss_count = 0 | PASSED | `reports/codex_session/calib_after.json`: all 0. |
| Calib-50 primary_failure_accuracy >= 0.48 | PASSED | `0.54`. |
| Set A (011, 012, 013): primary_failure must become CLEAN | PASSED | All three are `CLEAN` in `reports/codex_session/calib_after.json`. |
| Set B (11, 26, 39): target analyzer must reach status="fail" | PASSED | `retrieval_duplicate_chunks_11`: `RetrievalAnomalyAnalyzer=fail`; `citation_missing_26` and `quality_weak_grounding_39`: `CitationFaithfulnessAnalyzerV0=fail`. |
| Set B primary families remain unchanged | PASSED | `RETRIEVAL_ANOMALY`, `CITATION_MISMATCH`, `CITATION_MISMATCH`. |
| Heldout heldout_v0_1 primary_failure_accuracy >= 0.60 | PASSED | `0.667`. |
| On the 13 non-clean heldout cases: no primary_failure value may change | PASSED | `reports/codex_session/heldout_before.json` vs `heldout_after.json`: zero non-clean primary changes. Only `govrag-calib-seed-013` changed, from `CITATION_MISMATCH` to `CLEAN`. |

## Notes

- Engine warn-to-primary promotion was not removed.
- No golden labels, acceptable alternatives, production gates, or engine code were edited.
- Full `tests/test_analyzers/ -q` remains red with 20 failures in unrelated answer-quality, evidence-layer claim typing, triplet-verification, and version-validity tests; focused affected analyzer tests pass.
- Heldout-before was generated in a temporary copy of the current workspace with only the analyzer calibration code patch reversed, to preserve the pre-existing dirty workspace context while avoiding changes to the main worktree.
- Post-edit harness returned `warn` because the workspace contains protected/report changes, including files that were dirty before this implementation. The protected baseline gate itself passes.
