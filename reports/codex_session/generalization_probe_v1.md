# Generalization probe v1 — engine on fresh (out-of-distribution) data

**Date:** 2026-06-17
**Data:** 90 induced candidates from the starter seed batch (HotpotQA wiki +
deepset prompt-injections), produced by `scripts/induce_cases.py`. Labels are
by-construction (induced) or source-labelled.
**Method:** built a `RAGRun` per candidate (chunks + `final_answer` +
`cited_doc_ids`) and ran `diagnose()`; compared `primary_failure` to the label.

## Result

| expected | correct | most common engine output |
|---|---|---|
| CITATION_MISMATCH | **20/20** | CITATION_MISMATCH |
| INSUFFICIENT_CONTEXT | 3/20 | UNSUPPORTED_CLAIM, CITATION_MISMATCH |
| CLEAN | 3/20 | INCONSISTENT_CHUNKS, STALE_RETRIEVAL, INSUFFICIENT_CONTEXT |
| UNSUPPORTED_CLAIM | 1/20 | CITATION_MISMATCH (8/20) |
| PROMPT_INJECTION | 1/10 | UNSUPPORTED_CLAIM (9/10) |
| **Overall** | **28/90 = 0.31** | |

For comparison, the engine scores ~0.62 primary on its own (synthetic) Calib-50
fixtures. The drop to **0.31 on fresh data** is the key signal: prior numbers are
substantially overfit to the existing fixtures' construction.

## What's real vs. what's a caveat

**Caveats (don't over-read the single number):**
- Induced labels and a single domain (wiki); some induced CLEAN/INSUFFICIENT
  cases are arguable at the margin.
- This is a directional generalization probe, **not** a calibrated metric.

**Real, actionable patterns (robust to the caveats):**
1. **Over-firing on CLEAN (3/20).** The engine flags healthy answers as failures
   (`INCONSISTENT_CHUNKS`/`STALE_RETRIEVAL`/`INSUFFICIENT_CONTEXT`). High
   false-positive rate — a diagnosis tool that cries wolf. → **Task 17**
2. **PROMPT_INJECTION detected but not promoted (1/10).** The injection analyzer
   fires (logs "Prompt injection detected") but the decision policy routes primary
   to `UNSUPPORTED_CLAIM`. A security-relevant signal is buried. → **Task 18**
3. **Citation gate over-eager.** UNSUPPORTED_CLAIM cases are routed to
   CITATION_MISMATCH (same specificity-rank family as Task 16 / Task 3-v2).
4. **Bright spot:** CITATION_MISMATCH detection generalizes perfectly (20/20).

## Why this is good news, not bad

This is exactly the signal the tiny synthetic fixture set was hiding. The project
now has (a) a locked dataset, (b) a pipeline to add fresh cases, and (c) a probe
that measures real generalization. The path to "trustworthy" is now visible and
measurable instead of assumed. The fixes (Tasks 17, 18, and the citation
specificity work) are concrete and testable against this data.

## Reproduce

```
python scripts/pull_seed_intake.py     # internet required
python scripts/induce_cases.py
# then the probe snippet in this report (build RAGRun per candidate, diagnose).
```
