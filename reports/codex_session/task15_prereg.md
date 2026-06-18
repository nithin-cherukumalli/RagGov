# Task 15 pre-registration — incomplete-answer stage attribution (GROUNDING→GENERATION)

**Date:** 2026-06-18 · written BEFORE code.

## Scope (narrowed after instrumentation)
Two tests exist:
- `tests/test_pr5e_answer_quality.py::test_incomplete_answer_with_good_context_stage_generation`
  — currently **RED** (real failure, not xfail). Injects grounding `UNSUPPORTED_CLAIM`/GROUNDING +
  SufficiencyAnalyzer `pass`(sufficient). Asserts primary `UNSUPPORTED_CLAIM` + `root_cause_stage
  == GENERATION`. Does **not** assert selected_analyzer.
- `tests/test_analyzers/test_answer_quality_confidence_metadata.py::test_quality_incomplete_38...`
  — strict **xfail**. Additionally asserts `selected_analyzer == AnswerQualityAnalyzer`.
  `AnswerQualityAnalyzer` is NOT in the default suite (confirmed), so satisfying this needs wiring
  a new analyzer into the pipeline — broad blast radius. **Out of scope** this task; stays xfail.

This task fixes only the RED `test_pr5e` test via stage re-attribution. No new analyzer, no
primary-failure change, no decision-policy candidate change.

## Root cause
`root_cause_stage` is taken verbatim from `primary_result.stage` (engine.py ~404). A
`ClaimGroundingAnalyzer` `UNSUPPORTED_CLAIM` is stamped GROUNDING. But when the retrieved context
is *sufficient* (SufficiencyAnalyzer passes) and the answer still makes unsupported claims, the
fault is at GENERATION (the generator omitted/over-claimed despite adequate context), not retrieval
grounding.

## Change (one, narrow) — v2 after v1 revert
**v1 (reverted):** gating on "UNSUPPORTED_CLAIM + sufficient context" was too broad — it moved the
stage for generic over-claim cases too, dropping protected baseline 41→36. Reverted immediately.

**v2:** gate on the *incompleteness* signal specifically (the distinguishing feature of case 38):
query requests N enumerated items (`N steps/requirements/items`), the retrieved context contains
items 1..N, but the answer omits some. Reuse `AnswerQualityAnalyzer._answer_completeness_signals`
so the engine's definition of "incomplete" stays identical to the analyzer's. Gate:
`primary == UNSUPPORTED_CLAIM` AND `primary_result.analyzer_name == "ClaimGroundingAnalyzer"` AND
`root_cause_stage == GROUNDING` AND answer-incomplete-despite-context → `root_cause_stage =
GENERATION`. Primary failure unchanged; only the stage label moves. Generic over-claim/hallucination
unsupported cases (no requested-enumerated-items gap) are unaffected.

## Hard acceptance criteria
1. Protected baseline stays **41/46 GREEN**. *(revert)*
2. Calib scored primary stays **≥ 23/45** (primary unaffected; this is a stage-only change). *(revert)*
3. Probe overall primary accuracy unchanged (≥ 80/145). *(revert)*
4. No other test asserting `root_cause_stage` for UNSUPPORTED_CLAIM regresses; full
   `tests/test_analyzers` + `tests/decision_policy` + `tests/test_pr5e_answer_quality.py` green
   (modulo known pre-existing stale failures). *(revert)*

**Success:** `test_incomplete_answer_with_good_context_stage_generation` passes (RED→green).
`test_quality_incomplete_38...` remains xfail (documented: needs AnswerQualityAnalyzer wired in).
