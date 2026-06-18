# Task 21 result — NCV Jaccard duplicate-chunk over-firing — LANDED

**Date:** 2026-06-18. Prereg: `task21_prereg.md`.

## What changed
`src/raggov/analyzers/verification/ncv.py`: raised the context-assembly duplicate-chunk FAIL
threshold from `0.85` to `0.97` (named constant `_DUPLICATE_JACCARD_MIN`). A context-assembly
duplicate means the same chunk content was assembled more than once (≈identical); topical
overlap at 0.85–0.96 between distinct multi-hop passages no longer fails.

## Acceptance criteria
| # | Criterion | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Protected baseline | 41/46 | **41/46** | PASS |
| 2 | Calib scored primary | 23/45 (0.511) | **23/45 (0.511)** | PASS |
| 3 | NCV suite incl. identical-text TP | green | **16 passed** | PASS |
| 4 | Probe overall accuracy | 54/145 (0.372) | **57/145 (0.393)** | PASS (strict ↑) |
| 5 | CLEAN→INCONSISTENT_CHUNKS (Jaccard path) | 3 | **0** | MET |

CLEAN-correct rose 10/30 → **13/30**. Probe overall +3 (the 3 recovered Jaccard FPs).
Remaining CLEAN→INCONSISTENT_CHUNKS = 2, both the irreducible negation-path residuals from
Task 19 (need NLI), not this path.

## Tests
- `test_topically_overlapping_distinct_passages_do_not_fail_context_assembly` (precision guard,
  Jaccard ≈0.89 — would FAIL under the old 0.85 threshold, passes now).
- Existing identical-text TP `test_duplicate_chunks_fail_context_assembly` preserved.
