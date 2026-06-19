# Phase 2 increment 3 — grounded-clean gate (the real CLEAN-FP fix)

**Date:** 2026-06-18 · written BEFORE code.

## Hypothesis
Real CLEAN-FP (0.76) is death-by-a-thousand-cuts: long faithful answers trip low-tier
retrieval-health heuristics (stale/inconsistent/scope/sufficiency/citation). When an
**entailment-grade** grounding verdict shows the answer's claims are clean (none contradicted,
none unsupported), a single low-tier heuristic warning should NOT flip the verdict to a failure.
Gating on the NLI verdict (not the over-firing heuristic) is the principled, non-circular fix.

## Change (one, in the decision policy)
After the winner is selected, return CLEAN iff ALL of:
1. `winner.failure_type` ∈ a low-tier retrieval-health suppressible set
   {STALE_RETRIEVAL, INCONSISTENT_CHUNKS, SCOPE_VIOLATION, RETRIEVAL_ANOMALY, CITATION_MISMATCH,
   INSUFFICIENT_CONTEXT}; AND
2. no BLOCKING_DETERMINISTIC candidate exists (no injection/privacy/parser block); AND
3. the answer is **entailment-clean**: the ClaimGroundingAnalyzer result used an entailment-grade
   `verification_method` AND has no `contradicted` and no `unsupported` claim (abstain/skip ignored).

## Why safe (no-regression argument)
Condition 3 requires an entailment-grade verifier. In default/native mode (and the sandbox) the
grounding verifier is heuristic → no claim carries an entailment method → the gate NEVER fires →
default/native behavior is byte-identical. The gate only activates with an NLI/LLM verifier
configured (Kimi/Groq/local_nli). Dangerous direction guarded: if a real failure exists, the
entailment verdict flags it (contradicted/unsupported) → condition 3 false → no suppression.

## Hard acceptance criteria
1. Protected 43/46 unchanged. *(revert)*
2. Calib 23/45 unchanged. *(revert)*
3. Probe 80/145 (default) and real heldout 18/75 (default) unchanged. *(revert)*
4. `tests/decision_policy` + `tests/test_analyzers` green (modulo pre-existing stale fail). *(revert)*

**Success (measured by the user/S1 with Kimi):** real-heldout CLEAN-FP drops materially in NLI mode
with NO real failure-type row flipping to CLEAN. Unit tests prove: entailment-clean + low-tier
winner → CLEAN; a contradicted/unsupported entailment claim → NOT suppressed.
