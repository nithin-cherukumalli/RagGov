# Task 19 pre-registration — INCONSISTENT_CHUNKS over-firing

**Date:** 2026-06-18 · written BEFORE code.

## Root cause (established by instrumentation)

All 8 CLEAN→INCONSISTENT_CHUNKS probe false positives flow through
`has_suspicious_negation_pair` (`src/raggov/analyzers/retrieval/inconsistency.py`),
reached via the profile path: `RetrievalEvidenceProfilerV0._contradiction_candidates`
→ `NegationHeuristicContradictionDetector.compare_chunks` → `profile.contradictory_pairs`
→ `InconsistentChunksAnalyzer._from_profile` (warn) and NCV.

The heuristic fires when two chunks share **any one** non-stopword term that sits within
±5 tokens of a negation word ("not"/"however"/"in fact"...) in **either** chunk — with no
requirement that the two chunks actually disagree. On multi-hop HotpotQA/ALCE retrieval the
triggers are incidental lone tokens: `chlorine`/`world`/`which`/`his`/`has` near a "however"
or "not" that belongs entirely to one chunk's own unrelated sentence. The module docstring
already admits this ("False positives occur when negation is scoped to unshared terms").
`STOPWORDS` is tiny (~40 words) and misses pronouns/auxiliaries/relativizers, so function
words count as "shared terms".

This is a *contradiction-evidence* precision fix (the plan's mandate), not a query/passage
heuristic for a pipeline mode.

## Discriminator (TP vs FP)

- TP (must keep): "refund policy **applies** to hardware returns" vs "refund policy **does
  not apply** to hardware returns" — the negation window holds **≥2 shared content terms**
  (policy, hardware, returns) that the other chunk asserts.
- FP (must drop): a **single** incidental shared token near a negation; no shared proposition.

## Change (one, narrow)

Rewrite `has_suspicious_negation_pair` to require that a negation window in one chunk contains
**≥2 distinct shared *content* terms** (content = shared term, length ≥3, not in an expanded
FUNCTION_WORDS set, not numeric). Add `FUNCTION_WORDS` used only by this predicate. `terms()`,
`tokens()`, `STOPWORDS`, and `has_nearby_negation` signatures preserved (other modules import
them). No threshold/policy change elsewhere.

## Measured baseline (BEFORE — reproduced this session)

- Protected: 41/46 GREEN.
- Calib scored primary: 23/45 = 0.511 (no gold INCONSISTENT_CHUNKS in Calib).
- Probe overall: 43/145 (0.297, post-Task-18).
- Probe CLEAN→INCONSISTENT_CHUNKS false positives: **8** (of 30 CLEAN).

## Hard acceptance criteria

1. Protected baseline stays **41/46 GREEN**. *(revert trigger)*
2. Calib scored primary stays **≥ 23/45 = 0.511**. *(revert trigger)*
3. True-positive negation contradictions preserved:
   `has_suspicious_negation_pair` still **True** on the refund pair; the relevant analyzer/NCV
   tests (`test_negation_contradiction_detector_preserves_existing_heuristic`,
   `test_inconsistent_chunks_warns_on_nearby_negation_signals`,
   `test_duplicate_chunks_fail_context_assembly`) stay green. *(revert trigger)*
4. Probe overall accuracy does **not decrease** (≥ 43/145). *(revert trigger)*

**Success criterion:**
5. Probe CLEAN→INCONSISTENT_CHUNKS false positives drop materially: **8 → ≤2**.

Per the plan note: overall CLEAN-correct may rise only slightly (another over-firer, e.g.
CHUNKING_BOUNDARY_ERROR, can win the freed cases — that is Task 20). Track the **per-type FP
count** as the success measure, not overall CLEAN-correct. If 1–4 hold but 5 lands at 3–4,
keep only if a strict per-type improvement with zero TP regression; else revert.
