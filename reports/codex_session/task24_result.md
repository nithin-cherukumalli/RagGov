# Task 24 — list/short-answer false-CLEAN — DEFERRED (evidence-backed), no code change

**Date:** 2026-06-18. Decision: do not force an extraction change now. Reasons below.

## The remaining false-CLEAN slice (instrumented)
6 induced `INSUFFICIENT_CONTEXT → CLEAN` rows. Short/list answers; `SufficiencyAnalyzer`
(term-coverage, no LLM) returns sufficient=True; `HeuristicClaimExtractorV0` skips the short/list
answer → no grounding check → CLEAN.

Per-row answer-in-context check:
- Answer NOT in context (Arthur's Magazine, President Nixon, American, Jonathan Stark): a short
  answer ungrounded in retrieved context.
- Answer IN context (Delhi, alcohol): grounded short answer; gold `INSUFFICIENT` is about the
  QUESTION needing comparison/more, which a single entity can't satisfy.

## Why this is bounded, not a clean fix
1. **Wrong target label.** Gold is `INSUFFICIENT_CONTEXT`. The achievable extraction fix (make short
   answers verifiable) routes the ungrounded ones to `UNSUPPORTED_CLAIM`, not `INSUFFICIENT_CONTEXT`
   — so it yields **no exact-match probe gain**; it only converts silent-CLEAN to a different miss.
2. **No query context in the extractor.** `HeuristicClaimExtractorV0.extract_structured(answer)`
   has no access to the query, so the safe "query-conditioned short factual answer" discriminator
   (Codex's proposal) cannot be applied at extraction time without a larger signature/architecture
   change.
3. **CLEAN precision risk.** Codex's dry-run: a short-answer predicate fires on 11/30 expected-CLEAN
   rows, a comma-list predicate on 10/30. Broadening extraction blindly risks real clean short/list
   answers.
4. **Synthetic rows.** These are induced mutations; the only honest value is reducing dangerous
   silent-CLEAN, which does not move the headline metric.

## Decision
Defer. The safe version requires query-aware grounding (pass the query into extraction/verification
so a short answer is verified for support against context) — an architectural change worth doing
deliberately, not a quick extractor regex. Tracked for a future, properly-scoped task. Consistent
with the Task 22 stance: do not force a risky change with no clean gain.

No code/labels/gates changed.
