# Task 26 result — CLEAN→INSUFFICIENT_CONTEXT over-firing — REVERTED (gate-3 violation)

**Date:** 2026-06-18. Prereg: `task26_prereg.md`. Outcome: change implemented, measured, **reverted**
per pre-registered criterion. No code change kept.

## What was tried
Gated SufficiencyAnalyzer Pattern 5 (`missing_scope_condition`) on `_query_is_scope_general` —
fire only when the query has no disambiguating proper-noun entity. Prototype cleanly separated the
4 false positives (proper-noun multi-hop queries) from the genuine unit-test case ("What is the
sales tax?", 0 proper nouns).

## Why reverted
| Metric | Before | After change |
|---|---|---|
| CLEAN→INSUFFICIENT_CONTEXT (scope FP) | 5 | 1 (target met) |
| CLEAN-correct | 13/30 | 15/30 |
| **INSUFFICIENT_CONTEXT recall** | **4/30** | **1/30** ← **gate-3 violation** |
| Probe overall | 80/145 | 81/145 (+1 only) |
| Protected / Calib | 43 / 23 | 43 / 23 |

Pre-registered hard criterion 3 forbade any genuine `INSUFFICIENT_CONTEXT` true positive flipping
away. The fix dropped INSUFFICIENT recall 4→1: the **same incidental-state scope mechanism** that
produced the 4 CLEAN false positives was also producing 3 INSUFFICIENT-gold detections (right label,
spurious reason). Gating the query kills both. Net probe was only +1, not worth violating a
pre-registered gate, and I will not reinterpret "genuine" post-hoc. Reverted.

## Finding (the real shape of this bug)
`missing_scope_condition` is a coarse heuristic: it fires on a US-state name appearing via "in
<State>" in ANY retrieved chunk. Its CLEAN false positives and its INSUFFICIENT true positives are
**the same mechanism** — so query-level gating cannot separate them. A correct fix must validate that
the matched location actually **scopes the answer** (co-occurs with answer-bearing content that
addresses the query), not merely appears in some chunk. That is a more precise predicate (answer/
location co-occurrence) — deferred as a properly-scoped follow-up rather than forced here.

Consistent with the Task 22 / Task 24 stance: do not force a change that trades one error class for
another with no clean net diagnostic gain.
