# Protected Baseline Pin Decision — v0.1-alpha-public sprint, Day 1

**Date:** 2026-06-12
**Decision recorder:** Principal Evaluation Architect (Opus session)
**Status:** `C_REVERT_ROUTING` — pin unchanged, routing work stashed for Day 3 re-evaluation.

## Context

A Day 1 Task 1 (engineer-facing CLI) was scoped narrowly: add `--format text|json`
+ mandated calibration footer to `src/raggov/cli.py`; do not modify analyzers,
engine, models, or existing tests. The CLI work landed as specified.

During the same session, an additional set of routing changes was applied
beyond scope, touching 7 source files:

- `src/raggov/analyzers/answer_quality/analyzer.py`
- `src/raggov/analyzers/grounding/evidence_layer.py`
- `src/raggov/analyzers/grounding/verifiers.py`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
- `src/raggov/analyzers/verification/ncv_priority.py`
- `src/raggov/decision_policy_support.py`
- `src/raggov/engine.py`

These changes moved the protected common benchmark from 41/46 to 42/46 with a
different failure composition (4 newly-passing cases, 2 newly-failing cases).

## Question

Update the pinned protected baseline from `(41, 46)` to `(42, 46)` with the new
composition, or revert the routing changes?

## Evidence collected before deciding

### 1. Heldout split locked first

`evals/govrag_calib/splits/heldout_v0_1.{json,jsonl}` — 15 family-balanced cases
drawn from Calib-50, deliberately excluding cases worked on during the sprint
(007, 023, 030, 033, 034, 036, 037, 049). SHA256 content fingerprint:
`58261ecc57247f9fe80e114d126e914107c52bbe598acf851e4b616bc9b15c76`.

Family balance: answer_quality 2, citation 2, clean_pass 2, grounding 2,
retrieval 2, security_privacy 2, sufficiency 2, version_validity 1.

### 2. New-vs-old comparison on the locked heldout

| Metric | OLD code (routing stashed) | NEW code (routing applied) | Delta |
|---|---|---|---|
| `primary_failure_accuracy` | 0.600 | 0.600 | **0.000** |
| `stage_accuracy` | 0.400 | 0.400 | **0.000** |
| `false_clean_count` | 0 | 0 | 0 |
| `dangerous_clean_miss_count` | 0 | 0 | 0 |
| `human_review_miss_count` | 0 | 0 | 0 |
| `security_stage_miss_count` | 0 | 0 | 0 |
| Cases where actual primary/stage differed | — | — | **0 of 15** |

The routing changes produced **zero behavioral difference** on every heldout
case, including cases in the families the routing was designed to address
(citation, answer_quality, grounding, sufficiency).

### 3. Protected common benchmark

| Code state | Count | Composition vs alpha-clean pin |
|---|---|---|
| Alpha-clean pin (frozen earlier) | 41/46 | failures = {41, 38, 09, 08, 36} |
| OLD code (routing reverted) | 41/46 | failures = {20, 41, 38, 08, 36} — composition drift pre-exists this session |
| NEW code (routing applied) | 42/46 | failures = {23, 42, 08, 36} — +4 newly-passing, **-2 previously-passing**: `citation_phantom_23`, `quality_overconfident_weak_evidence_42` |

### 4. Calib-50 (no heldout)

Identical metrics OLD vs NEW: primary 0.480, stage 0.420, all safety counters 0,
`acceptable_nonclean_human_review_count=2`. Calibration status remains
`not_calibrated`, gating remains `False`.

## Decision

**`C_REVERT_ROUTING`.**

The routing edits do not earn a pin update because:

1. **No measured generalisation benefit.** Zero of fifteen unseen, family-balanced
   cases changed behaviour. The +1 common-benchmark improvement is therefore
   most consistent with overfitting to the specific failure fixtures the
   routing was targeted at, not with a generalisable diagnostic improvement.
2. **Real regressions on previously-passing cases.** `citation_phantom_23` and
   `quality_overconfident_weak_evidence_42` were passing under the pinned
   baseline composition and now fail. The +1 net delta hides a fidelity
   regression on those two cases.
3. **The discipline cost of accepting weak evidence is too high.** Updating the
   pin without a clear heldout signal sets the precedent that the pin moves
   with any net-positive change. The pin is the alpha integrity contract; it
   should move only with explicit, measured, recorded justification.
4. **Calib-50 shows no movement.** None of the safety, primary, or stage
   metrics on the Calib-50 evaluation changed. The routing therefore does not
   contribute to closing the remaining accuracy gap on the dataset we already
   call out as the evaluation target.

The routing changes are not "wrong" — they may well address real failure
patterns. They are **not yet measurable as improvements** under any frozen
evaluation slice. They are preserved in `git stash` (label
`DAY1_OPTION_C_old_code_check`) and should be re-introduced on Day 3 of the
sprint as a deliberate routing task with:

- per-rule unit tests,
- per-rule evidence of which heldout cases the rule was designed to fix,
- a re-run against the locked heldout showing per-family delta,
- a recorded acceptance criterion that the net delta on the heldout is
  non-negative and that no clean-pass heldout case regresses.

## What stays

- CLI changes (Day 1 Task 1 ✓): `src/raggov/cli.py` + `tests/test_cli/test_cli_smoke.py`.
- Earlier-session integrity fixes:
  `FailureType.CLAIM_EXTRACTION_FAILED` + STALE_RETRIEVAL human-review
  escalation in `src/raggov/models/diagnosis.py`,
  silent-CLEAN branch in `src/raggov/analyzers/grounding/support.py`,
  tests in `tests/test_analyzers/test_grounding.py` and
  `tests/test_models/test_human_review_escalation.py`.
- This decision file + `evals/govrag_calib/splits/heldout_v0_1.{json,jsonl}`.

## What the pin still says

`scripts/check_protected_baseline.py` still asserts `(41, 46)` with the
original failure composition. The current common benchmark **at OLD code**
matches the count but not the composition (pre-existing drift documented
earlier this session). That drift remains a separate `KNOWN_PIN_DRIFT`
state — not addressed by this decision — and is the proper subject of Day 2
work (lock the composition expectation alongside the count).

## Invariants preserved by this decision

- `production_gating_eligible = False`
- `calibration_status = not_calibrated`
- protected `false_clean_count = 0`, `false_security_count = 0`, `false_incomplete_count = 0`
- Calib-50 `false_clean_count = 0`, `dangerous_clean_miss_count = 0`, `human_review_miss_count = 0`
- pinned `(passed, total) = (41, 46)` unchanged
- no dataset labels changed
- no analyzer thresholds changed
- no launch gates flipped
- heldout split locked with content fingerprint
