# CLEAN-FP increment 2 — SufficiencyAnalyzer scope-condition guard

**Date:** 2026-06-22. Investigation-led (measured the over-fire, then pre-registered the guard +
criteria before finalizing).

## Problem
`missing_scope_condition` (sufficiency Pattern 5) flagged 8 truly-CLEAN heldout answers as
INSUFFICIENT_CONTEXT. It grabs any US-state name found by `\bin (State)\b` in ANY chunk — including
distractor chunks — and declares the answer scope-limited. Measured: **0 TP / 7 FP on the trusted
real gold**; on the synthetic probe it fires on the SAME answer as both TP and FP (coin-flip).

## Why not just disable it
One protected case depends on it: `sufficiency_missing_scope_15` ("What is the sales tax?" →
chunk "Sales tax in California is 7.25%" → answer "The sales tax is 7.25%", expected
INSUFFICIENT_CONTEXT). Full-disable drops protected to 41/46 (FAIL). So the fix must be surgical.

## The guard (one change, sufficiency.py Pattern 5)
Fire only when the ANSWER actually echoes the scoped statement — i.e. the answer genuinely
inherits the chunk's hidden scope. Concretely, for each `in <State>` match: skip if the state is
in the query or already in the answer; and require that the answer shares a content token
(len>3) with the ~80-char window around the location. The protected TP echoes "sales tax / 7.25%"
from the California clause (fires); the heldout FPs draw "Hawaii"/"Todd Phillips"/etc. from
unrelated clauses (suppressed).

## Hard acceptance criteria (revert on any fail) — ALL MET
| criterion | required | result |
|-----------|----------|--------|
| protected baseline | pass (incl. sufficiency_missing_scope_15) | **PASS 42/46** |
| Calib native/default | 23/45 | **23/45 / 23/45** |
| probe native | >= 82/145 | **85/145** |
| heldout CLEAN-FP | down | **0.65 -> 0.52** (8 Sufficiency FP -> 2) |
| heldout detection | not dropped | **0.62 held** |
| heldout exact | not dropped | **0.36 -> 0.44** |
| sufficiency/scope tests | green (modulo pre-existing) | only pre-existing subtle_plausible_hallucination_03 red (fails on HEAD too) |

## Net
CLEAN-FP 0.65 → 0.52, exact 0.36 → 0.44, zero regressions. Combined with increment 1
(InconsistentChunks), CLEAN-FP has gone 0.76 → 0.52 on real data via two surgical, fully-guarded
analyzer fixes — no NLI, no benchmark gaming.
