# Failed approaches ledger — do not repeat these

A single place recording every change that was attempted and **reverted**, with
the specific mistake and the guardrail it teaches. Read before touching the
engine. "Reverted" here is success of the discipline, not failure of the project:
each pre-registered criterion caught a regression before anything false shipped.

| # | Attempt | What was tried | Why it failed | Lesson / guardrail |
|---|---|---|---|---|
| A | **Warn→primary promotion removal** (historical, pre-this-session) | Removed the warn-to-primary promotion in the engine | Calib false-clean count regressed (analyzers under-fired without it) | Don't remove a compensating mechanism until the analyzers it compensates for are fixed. |
| B | **Task 3 v1 — CITATION_MISMATCH** | Swapped the citation gate (`_answer_has_specific_value` → `_cited_doc_is_retrieved`) | Analyzer-only change; engine decision policy still routed to UNSUPPORTED_CLAIM; new gate was *less* specific → false-fired | An analyzer change without a **decision-policy rank** change is half a fix. Keep specificity ≥ the gate you replace. |
| C | **Task 4 v1 — RETRIEVAL_DEPTH_LIMIT** | Predicate parsed cardinal numbers ("five","six") from the **query string** | False-fired on CLEAN, broke unrelated cases; heldout 0.733→0.667 | **Never** use query-string/text heuristics for a pipeline failure mode. Read `run.metadata` / chunk metadata only. |
| D | **Task 5 v1 — RERANKER_FAILURE** | Registered analyzer but no decision-policy promotion | Engine never selected it; target case unchanged; also no golden data exists | Analyzer + decision-policy rank together, and only for failure types with real data. |
| E | **Task 17 attempt 1 — STALE over-firing** | `skip` the whole version-validity analyzer when no structured temporal signal | Protected baseline broke (version_validity 0/5); Calib 0.511→0.356; engine fell through to INCOMPLETE_DIAGNOSIS (probe CLEAN 3→0) | The analyzer is load-bearing. **Don't disable a whole analyzer** to suppress one output; the report it produces is consumed downstream and pinned by the baseline. |
| F | **Task 17 attempt 2 — STALE over-firing** | Removed the `(YYYY)`/"as of YYYY" → `issue_date` text inference | Probe improved (CLEAN 3→5, FP 5→1, Calib held) BUT protected case `version_stale_not_cited_32` regressed (40/46) | The over-firing heuristic is the **same one a pinned case depends on**. You can't delete a shared heuristic; you must split its two behaviours. |

## The cross-cutting lessons (apply to every future engine change)

1. **Pre-register hard criteria before code.** Always. Protected ≥ 41/46, Calib
   not regressed, named true-positives preserved, safety counters 0.
2. **Analyzer change ⇒ check the decision policy too.** Firing in isolation ≠
   being selected by the engine.
3. **No query-string / passage-text heuristics for pipeline failure modes.** Read
   structured metadata. (Task 17's whole problem is a text heuristic.)
4. **Don't disable load-bearing components.** Narrow the *behaviour*, not the
   component.
5. **A shared heuristic with two consumers needs splitting, not deleting.** (The
   Task 17 root cause.)
6. **Measure on fresh data, not just fixtures.** The 0.62→0.31 gap only showed up
   on the generalization probe.
7. **Revert on any failed criterion. Keep the pre-registration as the record.**
   Two reverts beat one false "fix".

## Open redesign

**Task 17-v3** — split *relative recency* (older competing version present →
legitimately stale; the `version_stale_not_cited_32` case) from *absolute age vs
an assumed "now"* (lone year in prose → the false positives). See
`task17_result.md`. Needs its own pre-registration (next).
