# Tasks 3-v2 / 4-v2 / 5-v2 — Feasibility Check (BLOCKER)

**Date:** 2026-06-17
**Author:** automated pre-registration feasibility pass (read-only; no code, test, or label changes)
**Verdict:** **All three v2 tasks are BLOCKED as written.** Do not implement until a
human reconciles the issues below. This note exists to prevent a third doomed
attempt (cf. the v1 revert `e759a12` and `reports/codex_session/tasks_3_4_5_result.md`).

## Why this check was run

Per the Prime Directive, pre-registration must precede code, and predicates must
never be tuned to specific fixtures. Before pre-registering any v2 task I checked
that (a) the cases named in the acceptance criteria carry the golden labels the
criteria assume, and (b) the pipeline-introspection signals the predicates require
actually exist in the fixtures. Both checks failed.

## Finding 1 — Case IDs in the v2 criteria no longer match the dataset goldens

The v2 criteria (and the v1 result doc) reference bare case numbers
(004, 011, 012, 019, 020, 023, 025, 027, 029, 030). The live dataset
`evals/govrag_calib/govrag_calib_150.jsonl` (52 rows; the Calib-50 + 3 heldout)
was relabeled/renumbered since the v1 attempt (2026-06-15). The `gc-0NN` IDs now
carry different golden labels than those numbers did in `tasks_3_4_5_result.md`.

| case | label in v1 result doc / v2 criteria | current golden in govrag_calib_150 |
|---|---|---|
| 025 | expected `UNSUPPORTED_CLAIM` (v2: "has no citations") | **`CITATION_MISMATCH`, 2 citations** |
| 029 | expected `CITATION_MISMATCH` | **`UNSUPPORTED_CLAIM`** |
| 030 | expected `CITATION_MISMATCH` | **`CONTRADICTED_CLAIM`** |
| 004 | expected `RETRIEVAL_DEPTH_LIMIT` | **`CLEAN`** |
| 019 | expected `RETRIEVAL_DEPTH_LIMIT` (implied) | **`UNSUPPORTED_CLAIM`** |
| 011 | expected `CLEAN` | **`STALE_RETRIEVAL`** |
| 012 | expected `CLEAN` | **`STALE_RETRIEVAL`** |
| 023 | expected `UNSUPPORTED_CLAIM` | **`CONTRADICTED_CLAIM`** |
| 020 | expected `RERANKER_FAILURE` | **`UNSUPPORTED_CLAIM`** |

Consequence: implementing Task 3-v2 to "flip 029, 030 to CITATION_MISMATCH" would
push two cases **away** from their current goldens (a regression and a
fixture-tuning violation). Task 4-v2's "004, 019 → RETRIEVAL_DEPTH_LIMIT / 011,
012 stay CLEAN" targets cases whose goldens are now CLEAN / UNSUPPORTED_CLAIM /
STALE_RETRIEVAL. The criteria cannot be satisfied honestly against current data.

Per Prime Directive rules 2 and 3, the goldens must **not** be edited to fit the
criteria, and the criteria must **not** be silently loosened. This is a
human-adjudication item: re-pin each v2 task to the correct current `gc-0NN` IDs
(or confirm a canonical Calib-50 snapshot), then re-write the acceptance criteria.

## Finding 2 — The real target cases exist, but the required signals are missing

Read against the *correct* current cases:

**RETRIEVAL_DEPTH_LIMIT** goldens are `gc-041`, `gc-042`, `gc-047`
(`expected_retrieval_issue = "depth_limit"`). But:
- There is **no `top_k`, no `k_floor`, no reranker field** at run level. `RAGRun.metadata`
  is a generic dict and is **empty** for these fixtures; chunk `metadata` is `{}`.
- Chunks carry only `rank` and `score` — no configured floor or top_k to compare
  against, so there is no pipeline-agnostic saturation signal.
- The only "depth" marker is `expected_retrieval_issue` itself, which is a **golden
  output label**, not a pipeline-introspection input. Using it as a predicate would
  be circular (reading the answer key) — exactly the heuristic class the Prime
  Directive forbids.
- `gc-047` is a placeholder: notes say `"TODO: Populate with real depth limit case."`

→ **Task 4-v2 hits its own pre-registered deferral clause** ("If those signals are
not present, this task is DEFERRED"). **DEFER.**

**RERANKER_FAILURE**: **zero** cases in the dataset carry this golden label, and
there is **no reranker metadata** anywhere (no `rerank_score`, no reranker fields).
The required end-to-end test ("case 020 → RERANKER_FAILURE" plus a negative case
with reranker metadata absent) cannot be constructed or validated.

→ **Task 5-v2 is not viable against current data. DEFER** until reranker-bearing
fixtures exist.

**CITATION_MISMATCH** goldens are `gc-024`, `gc-025`, `gc-051`, `gc-052`. The
failure mode *is* represented, but the v2 gate hypothesis does not fit:
- `gc-024`: cited docs (`doc-99`, `doc-47`) are **not in** the retrieved set
  (`doc-12/15/20`) → the cited doc is *not retrieved*.
- `gc-025`: citations are `...-WRONG` variants whose IDs differ from the retrieved
  docs → again *not bound* to a retrieved doc.
- `gc-051` (cited doc retrieved) and `gc-052` (no citations) are **placeholders**
  (`todo-cit-doc-1`, `todo-missing-cit-doc-1`).

The v2 gate fires only when "a citation points to a **retrieved** doc AND the cited
content does not support the answer" — but the two non-placeholder goldens
(024, 025) are *cited-doc-not-retrieved* cases, which that gate would miss. The
gate needs to cover the "cited doc absent from retrieval" pattern too.

→ **Task 3-v2 is mis-specified** against current data (wrong target IDs + a gate
that doesn't match the real cases' shape). Re-scope before pre-registration.

## Recommended next actions (for human adjudication — not done here)

1. **Re-pin case IDs.** Decide the canonical Calib-50 snapshot and rewrite each v2
   task's acceptance criteria against current `gc-0NN` goldens. Do **not** edit
   goldens to match the old criteria.
2. **Task 4-v2 → DEFER** (no introspection signal in fixtures). If depth-limit
   detection is wanted, first add `top_k`/`k_floor` to the run schema and populate
   `gc-041/042/047` fixtures with real values — but note the v2 "out of scope" rule
   forbids adding metadata *just to make the task land*; this needs an explicit
   schema decision.
3. **Task 5-v2 → DEFER** (no RERANKER_FAILURE goldens, no reranker metadata).
4. **Task 3-v2 → re-scope** to the real CITATION_MISMATCH cases (gc-024, gc-025;
   replace placeholders gc-051/052 first), and broaden the gate to include
   "cited doc not in retrieved set." Then pre-register and require the
   decision-policy specificity-rank change alongside the analyzer change (the
   lesson from v1).

## Impact on Task 11

Task 11's precondition is "Tasks 1–5 must land." Given the above, Tasks 3/4/5-v2
**cannot land as written**. Task 11 remains correctly blocked. No engine cleanup
should proceed until the dataset/criteria reconciliation above is resolved.

## Guardrails

Read-only investigation. No production code, tests, fixtures, or golden labels were
modified. No v2 implementation was attempted.
