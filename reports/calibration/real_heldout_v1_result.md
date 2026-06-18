# Real heldout v1 — first honest, non-synthetic generalization number

**Date:** 2026-06-18. Source: `fresh_intake_v1.jsonl` (user pull), mapped to
`evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`. **Locked — never tune on it.**

## Composition (75 rows, 0 overlap with training/probe/seeds)
- 25 RAGTruth `contradicted` → CONTRADICTED_CLAIM (provisional, **needs human audit**).
- 50 HotpotQA/ALCE reference answers → CLEAN (provisional; reference answers are faithful by
  construction, so CLEAN is a reasonable label).
- No real UNSUPPORTED/INSUFFICIENT rows (RAGTruth returned 0 baseless QA rows) — heldout covers
  **2 of 3 supported types**.

## The number (current committed engine)
| Set | default | native |
|---|---|---|
| **Real heldout v1 overall** | **18/75 = 0.24** | 19/75 = 0.253 |
| CLEAN | 12/50 = **0.24** | 13/50 = 0.26 |
| CONTRADICTED_CLAIM | 6/25 = 0.24 | 6/25 = 0.24 |

## What this means — read it straight
1. **The synthetic probe (0.55) overstated real performance by ~2.3×.** The honest
   production-generalization number is **≈0.24**, not 0.55. This is exactly why the plan demanded a
   real heldout; the synthetic number was not trustworthy.
2. **The dominant real-world problem is CLEAN over-firing.** Only 24% of genuinely-faithful
   reference answers are correctly called CLEAN — the engine raises a FALSE failure on ~76% of clean
   real answers. The synthetic probe hid this (CLEAN was 43% there) because real answers are longer,
   multi-fact, entity/date/citation-rich, which trips the lexical heuristics far more than the
   short synthetic clean cases did.
3. **Contradiction recall is ~24%** — the known native-heuristic limitation (Task 22). Confirms the
   NLI tier (Phase 2) is required, not optional, for trust.

## Honest caveats
- CONTRADICTED labels are RAGTruth-migrated heuristics; the 25 are flagged for human audit. Even
  granting them, recall is 6/25.
- CLEAN labels assume reference answers are faithful (reasonable, but a few may themselves be
  imperfect — audit a sample).
- 2-type coverage only; UNSUPPORTED/INSUFFICIENT generalization still unmeasured (need more data).

## Plan impact (this reorders priorities)
- The #1 trust problem on real data is **CLEAN false-positive rate**, not contradiction. Precision
  work must now be measured on THIS heldout, not the synthetic probe.
- Phase 2 (NLI tier) and Phase 3 (calibration) are validated as necessary. Add an explicit Phase-2.5
  target: **drive real-heldout CLEAN precision up** by making the lexical analyzers defer to the
  NLI/entailment verdict (a clean answer that entails its context should not be flagged).
- THE metric from now on is real-heldout accuracy (≈0.24 today) + CLEAN false-positive rate, tracked
  alongside the synthetic probe (no longer the headline).
