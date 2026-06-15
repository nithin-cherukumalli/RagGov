# Tasks 3, 4, 5 Result — REVERTED

**Status:** `REVERTED` — none of the three changes earned their landing per pre-registered acceptance criteria.
**Date:** 2026-06-15
**Pre-registrations:** `reports/codex_session/task{3,4,5}_prereg.md`

## What was tried

| Task | Change | Files |
|---|---|---|
| 3 | Citation gate replaced `_answer_has_specific_value` with `_cited_doc_is_retrieved` | `src/raggov/analyzers/citation_faithfulness/analyzer.py` |
| 4 | New `RetrievalDepthLimitAnalyzer` registered in Layer 3 | `src/raggov/analyzers/retrieval/depth.py`, `src/raggov/engine.py` |
| 5 | New `RerankerFailureAnalyzer` registered in Layer 3 | `src/raggov/analyzers/retrieval/reranker.py`, `src/raggov/engine.py` |

## What we measured

### Hard invariants

| Invariant | Required | After change | Verdict |
|---|---|---|---|
| Protected pin `(42, 46)` | unchanged | unchanged | ✅ |
| Calib-50 `false_clean_count` | 0 | 0 | ✅ |
| Calib-50 `dangerous_clean_miss_count` | 0 | 0 | ✅ |
| Calib-50 `human_review_miss_count` | 0 | 0 | ✅ |
| **Heldout primary accuracy** | **≥ 0.733** | **0.667** | **❌ VIOLATED** |
| Calib-50 primary accuracy | ≥ 0.62 | 0.54 | ❌ VIOLATED |

### Per-task acceptance criteria

**Task 3** — required ≥2 of {025, 029, 030} flip to `CITATION_MISMATCH`:

| Case | Pre-change | Post-change | Expected | Verdict |
|---|---|---|---|---|
| 025 | `CITATION_MISMATCH` | `CITATION_MISMATCH` | `UNSUPPORTED_CLAIM` | unchanged (still wrong) |
| 029 | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | `CITATION_MISMATCH` | unchanged |
| 030 | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | `CITATION_MISMATCH` | unchanged |

→ 0 of 2 required flips. **FAILED.**

**Task 4** — required cases 004, 019 → `RETRIEVAL_DEPTH_LIMIT` AND no false fire on cases with ≥5 chunks and substantive alignment:

| Case | Pre-change | Post-change | Expected | Verdict |
|---|---|---|---|---|
| 011 | `CLEAN` | `RETRIEVAL_DEPTH_LIMIT` | `CLEAN` | ❌ false positive on CLEAN case |
| 012 | `CLEAN` | `RETRIEVAL_DEPTH_LIMIT` | `CLEAN` | ❌ false positive on CLEAN case |
| 023 | `UNSUPPORTED_CLAIM` | `RETRIEVAL_DEPTH_LIMIT` | `UNSUPPORTED_CLAIM` | ❌ broke a correct case |
| 027 (heldout) | `CITATION_MISMATCH` | `RETRIEVAL_DEPTH_LIMIT` | `CITATION_MISMATCH` | ❌ broke heldout correctness |

→ 4 silent regressions, the predicate does not generalise. **FAILED.**

**Task 5** — required case 020 → `RERANKER_FAILURE`:

| Case | Pre-change | Post-change | Expected | Verdict |
|---|---|---|---|---|
| 020 | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | `RERANKER_FAILURE` | unchanged |

→ Target case did not flip. The analyzer fires in isolation (per code-tracing claim) but the engine routing does not promote it. **FAILED.**

## Why this is a clean revert, not a failure of discipline

The pre-registrations existed precisely to catch this. The heldout-stability invariant ("never regresses below 0.733") flagged a real regression caused by Task 4. Tasks 3 and 5 produced no movement at all — the implementations exist but are dead at the engine layer.

Reverting now is the same posture the project took for the warn-promotion attempt
(`reports/forensics_v0_1_warn_promotion_result.md`): when criteria fail, revert.
This preserves Calib-50 0.62 and Heldout 0.733, both earned with discipline.

## Root causes of the failures

### Task 3 — wrong root-cause hypothesis

The forensics had labelled cases 029, 030 as "missing_analyzer" (the right
failure type is never produced by any analyzer). The Task 3 change only
altered a gate on `CitationFaithfulnessAnalyzerV0`. But the analyzer was
already producing `warn` on those cases — the gate change made it produce
`fail`, but the engine's decision policy still routed elsewhere
(`ClaimGroundingAnalyzer.UNSUPPORTED_CLAIM` outranks). The fix would need
to be at the decision-policy specificity rank, not the analyzer gate.

Additionally, the new gate (`_cited_doc_is_retrieved`) is *less* specific
than the old (`_answer_has_specific_value`) — it fires whenever a citation
points to a retrieved doc, which is true for many UNSUPPORTED_CLAIM cases
too. Case 025 (no citations) was unaffected because the gate requires
citations to exist; it stays wrong for an unrelated reason.

### Task 4 — predicate not pipeline-agnostic

"retrieved ≤ k_floor AND query cardinal N > retrieved count" is a textual
heuristic on the query string ("five", "six"), not a property of the
retrieval pipeline. It fires whenever a query happens to mention a small
number — including clean cases like 011, 012 (CLEAN with small numeric
queries) and 023 (different failure type entirely). The pre-registration
warned against this: "Pre-registration must define the predicate in
dataset-independent terms (no thresholds tuned to 004/019)." The
implementation drifted toward query-pattern matching, which is exactly
what we ruled out.

### Task 5 — engine-level routing not addressed

Even if the analyzer fires in unit-test isolation, the engine's decision
policy needs a specificity rank that promotes `RERANKER_FAILURE` over
the existing analyzers that also fire on case 020. The Task 5 work
landed the analyzer but did not touch decision policy.

## What was preserved

Last-good state (commit `a17f75c` head):
- Protected pin `(42, 46)` GREEN
- Calib-50 primary `0.62`, stage `0.54`
- Heldout primary `0.733`, stage `0.533`
- All safety counters 0
- Tasks 1 + 2 (parser false-fire fix + STALE specificity rank) intact

## Reformulated tasks for next pass

**Task 3-v2** must include the **decision-policy specificity rank**
required to actually promote `CITATION_MISMATCH` on cases 029, 030. The
gate change alone is insufficient; pre-registration must enumerate both
the analyzer change and the rank change.

**Task 4-v2** must use only pipeline-introspection signals (e.g., a
`top_k` value in `run.metadata` or `chunk.metadata`, or chunk-rank
saturation patterns), never query-string number parsing. If
pipeline-introspection signals are not present in the fixtures, the task
should be deferred — there is no pipeline-agnostic way to detect depth
limit without that signal.

**Task 5-v2** must include both the analyzer AND the decision-policy
rank promotion for `RERANKER_FAILURE` to outrank `UNSUPPORTED_CLAIM`
when reranker metadata is present.

All three v2 tasks must add **direct end-to-end tests** that run
`DiagnosisEngine(...).diagnose(run)` and assert the resulting
`primary_failure`, not just analyzer-isolation tests. The isolated tests
in this attempt all passed; only the engine-level assertion would have
caught the routing failures.
