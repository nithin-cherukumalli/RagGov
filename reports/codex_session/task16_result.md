# Task 16 result — case-41 permission-contradiction specificity — LANDED (correctness fix)

**Date:** 2026-06-18. Prereg: `task16_prereg.md`.

## What changed
`src/raggov/decision_policy_support.py`: extended `_claim_has_textual_contradiction` with a third
disjunct — a claim asserting BROAD/unrestricted permission ("wear/do/use … anything", "anything you
want", "no restrictions/dress code") against evidence stating an explicit RESTRICTION ("no <X>",
"not allowed", "prohibited", "must/may not", "only … allowed", "required to"). Added `import re`.

## Why this is safe where Task 22 was not
Task 22's rows had matching values / no real conflict (promoting them regressed Calib + invented
false contradictions). Case 41 is a genuine, human-obvious contradiction ("Policy: No blue shirts."
vs "you can wear anything you want") that the heuristic missed only due to narrow permission
vocabulary. The new predicate fires on **0 of 145** probe rows and only matters for claims the
verifier already labeled contradicted — minimal blast radius.

## Acceptance criteria
| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Protected baseline | **43/46 effective, check PASS** | PASS |
| 2 | Calib scored primary | **23/45 (0.511)** unchanged | PASS |
| 3 | Probe overall + no new false CONTRADICTED | **80/145**, 0 new false contradictions | PASS |
| 4 | Suites green | `test_analyzers`+`decision_policy` 592 passed, 3 xfailed, 1 pre-existing stale | PASS |

**Result:** case 41 primary `UNSUPPORTED_CLAIM → CONTRADICTED_CLAIM` (the correct, more-specific
label). Added `test_permission_contradiction_e2e.py` (TP + precision guard).

## Honest scope note
The strict-xfail `test_quality_ignores_context_41...` still xfails: it additionally asserts
`root_cause_stage == GENERATION` and `selected_analyzer == AnswerQualityAnalyzer`, which (like the
`_38` test) require wiring `AnswerQualityAnalyzer` into the default suite — a separate broad change.
This task fixed the **primary-failure specificity** (the part achievable safely in native mode); the
stage/analyzer assertions remain pending that wiring. This is a permanent correctness improvement,
not a test-pass: it does not flip its own xfail test, but it makes the diagnosis correct.
