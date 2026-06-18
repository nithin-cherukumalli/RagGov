# Task 19 result — INCONSISTENT_CHUNKS over-firing — LANDED

**Date:** 2026-06-18. Prereg: `task19_prereg.md`.

## What changed
`src/raggov/analyzers/retrieval/inconsistency.py`: rewrote `has_suspicious_negation_pair`
(the shared chokepoint used by the profile path, `NegationHeuristicContradictionDetector`,
and NCV). A pair is now flagged only when the **same multi-term proposition** is negated in
one chunk and asserted in the other:
1. strong negations only (`not/never/cannot/no longer`) — discourse markers
   (`however/in fact/but actually`) dropped as they drove cross-topic FPs;
2. the negation's ±5 window must hold **≥2 shared *content* terms** (new `FUNCTION_WORDS`
   filter excludes pronouns/auxiliaries/relativizers; `STOPWORDS` untouched for other consumers);
3. that cluster must appear **asserted** (co-located, negation-free) in the other chunk.

## Acceptance criteria
| # | Criterion | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Protected baseline | 41/46 | **41/46** | PASS |
| 2 | Calib scored primary | 23/45 (0.511) | **23/45 (0.511)** | PASS |
| 3 | TP negation/NCV tests green | green | **green** (4 named + full suite 112) | PASS |
| 4 | Probe overall accuracy | 43/145 (0.297) | **47/145 (0.324)** | PASS (strict ↑) |
| 5 | CLEAN→INCONSISTENT_CHUNKS FPs (negation path) | 8 | **2** (target ≤2) | MET |

CLEAN-correct rose 4/30 → **8/30**. Probe overall +4 (all CLEAN recovery).

## Honest residuals
- **2 irreducible negation-path FPs** (House of Anubis TV series; a heavy-metal-band pair).
  These are *lexically identical* in structure to the protected true positive
  ("policy applies" vs "policy does **not** apply" ≈ "episodes aired" vs "episodes **never**
  aired in the US"). Separating them needs NLI/entailment, not lexical rules; forcing them
  down would break the refund TP. Documented as the heuristic's boundary (the module already
  declares itself "NOT NLI").
- **2 further CLEAN→INCONSISTENT_CHUNKS** now surface from a *different* mechanism —
  NCV's `context_assembly` **Jaccard duplicate-chunk** fallback firing on genuinely
  near-identical ALCE passages (0.90/0.96). Previously masked by the negation path. This is a
  separate root cause, out of Task 19's scope → filed as **Task 21** (Jaccard duplicate
  over-firing). Total INCONSISTENT-on-CLEAN is therefore 4 (2 negation + 2 Jaccard); the
  negation mechanism this task scoped hit the ≤2 target.

## Tests
- Added `test_inconsistent_chunks_ignores_incidental_single_token_negation` (precision guard,
  the dominant FP class). All prior TP tests preserved.
