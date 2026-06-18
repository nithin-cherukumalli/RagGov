# Task 15 result — incomplete-answer stage attribution — LANDED (v2)

**Date:** 2026-06-18. Prereg: `task15_prereg.md`.

## v1 reverted, v2 landed
- **v1 (reverted):** gating stage re-attribution on "UNSUPPORTED_CLAIM + sufficient context" was
  too broad — protected baseline dropped 41→36 (generic over-claim cases moved stage too).
  Reverted within minutes; kept the prereg as the record.
- **v2 (landed):** gate on the *incompleteness* signal only — query requests N enumerated items,
  the retrieved context contains 1..N, but the answer omits some. Reuses
  `AnswerQualityAnalyzer._answer_completeness_signals` so the engine's definition of "incomplete"
  matches the analyzer's.

## What changed
`src/raggov/engine.py`: added `_answer_incomplete_despite_context(run)` and a narrow stage rule —
if primary is `UNSUPPORTED_CLAIM` from `ClaimGroundingAnalyzer` at GROUNDING and the answer is
incomplete-despite-context, set `root_cause_stage = GENERATION`. **Primary failure unchanged.**
`scripts/check_protected_baseline.py`: removed `quality_incomplete_38` from `EXPECTED_FAILURES` and
bumped the effective count 42→43, because the case is now genuinely fixed (documented in-file,
matching the existing `retrieval_irrelevant_plausible_09` precedent). No golden labels changed.

## Acceptance criteria
| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Protected baseline | **43/46 effective, check PASS** (case 38 fixed, no green regressed) | PASS (improvement) |
| 2 | Calib scored primary | **23/45 (0.511)** unchanged | PASS |
| 3 | Probe overall primary accuracy | **80/145 (0.552)** unchanged (stage-only) | PASS |
| 4 | No stage-assertion regressions | `tests/test_analyzers`+`decision_policy`+`test_pr5e` → 597 passed, 3 xfailed, 1 pre-existing stale fail | PASS |

**Success:** `test_incomplete_answer_with_good_context_stage_generation` RED→green; protected case
`quality_incomplete_38` now passes (stage GENERATION).

## Scope note
The strict-xfail `test_quality_incomplete_38_has_generation_stage_candidate_if_supported` still
xfails — it additionally requires `selected_analyzer == AnswerQualityAnalyzer`, which needs that
analyzer wired into the default suite (a broad pipeline change). Left xfail; the *engine* now
attributes the stage correctly regardless of which analyzer is selected. Wiring AnswerQualityAnalyzer
is a separate, larger task to scope with full probe/Calib/protected validation.
