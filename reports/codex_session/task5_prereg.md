# Task 5 Pre-Registration — RERANKER_FAILURE detection

**Date:** 2026-06-15  
**New file:** `src/raggov/analyzers/retrieval/reranker.py`

---

## Recognized metadata keys (explicit path)

The analyzer inspects `chunk.metadata` for the following key pairs:

| Meaning | Key A | Key B (synonym) |
|---------|-------|----------------|
| Pre-rerank rank | `initial_rank` | — |
| Pre-rerank score | `pre_rerank_score` | — |
| Post-rerank rank | `final_rank` | — |
| Post-rerank score | `rerank_score` | — |

At least one key from each pair must be present in at least one chunk's metadata for the explicit path to activate.

---

## "Rerank metadata present" definition

"Rerank metadata" includes two forms:

1. **Explicit**: one or more retrieved chunks expose `initial_rank`/`pre_rerank_score` and `final_rank`/`rerank_score` in `chunk.metadata`.
2. **Implicit/positional**: the ordering of `run.retrieved_chunks` encodes rank (index 0 = rank 1, index 1 = rank 2, etc.), which is always available when `len(run.retrieved_chunks) > 1`.

The analyzer uses whichever form is available. If the retrieved list has 0 or 1 chunk, it skips (insufficient ordering evidence).

---

## Dataset-independent predicate

### Path 1 — Explicit rerank metadata

Fire RERANKER_FAILURE when:
1. At least one chunk's `metadata` contains both a pre-rerank signal (`initial_rank` or `pre_rerank_score`) and a post-rerank signal (`final_rank` or `rerank_score`).
2. The chunk most textually aligned with the answer (highest token overlap with `run.final_answer`) was upranked (`final_rank < initial_rank` or `rerank_score > pre_rerank_score`), AND
3. A different chunk that was downranked (`final_rank > initial_rank`) has higher token overlap with the query than the upranked chunk.

### Path 2 — Positional/grounding path (implicit ordering)

Fire RERANKER_FAILURE when:
1. `grounding_evidence_bundle` is available in `run.metadata` (from `ClaimGroundingAnalyzer`).
2. At least one claim has `verification_label = CONTRADICTED`.
3. The contradicting chunk for that claim is NOT at position 0 in `run.retrieved_chunks` (i.e., it was not the top-retrieved chunk).
4. The chunk at position 0 does NOT contradict the same claim (indicating the top chunk displaced the correcting chunk).
5. The score difference between position-0 chunk and the contradicting chunk is ≤ `rank_score_delta` (default 0.05), indicating the reranker made a marginal ordering decision.

Condition 5 avoids firing when the top chunk is clearly more relevant by score (large gap suggests retriever ordering was intentional).

---

## Case 020 trace

- c1 (index 0, score 0.91): "Sev-2 incidents use S2-RED" — does NOT contradict claim-1 ("Sev-1 → S2-RED")
- c2 (index 1, score 0.90): "Sev-1 incidents use S1-BLACK" — CONTRADICTS claim-1
- claim-1: `verification_label = CONTRADICTED`, contradicting_chunk_ids = ["c2"]
- Path 2 conditions: c2 is at index 1 (not 0) ✓, c1 does not contradict claim-1 ✓, |0.91 − 0.90| = 0.01 ≤ 0.05 ✓
- → Fire RERANKER_FAILURE

---

## "Skips cleanly" guarantee

- If `len(run.retrieved_chunks) <= 1`: skip (no ordering evidence).
- If no `grounding_evidence_bundle` and no explicit rerank metadata: skip.
- If `grounding_evidence_bundle` present but no `verification_label = CONTRADICTED` claim: skip.
- Path 2 requires a genuine grounding contradiction, which limits false positives to cases that genuinely have contradicted claims with a specific rank pattern.

---

## Acceptance criteria mapping

- Case 020 → RERANKER_FAILURE via Path 2 ✓
- Cases without contradicted claims → skip ✓
- Protected pin cases: none of the protected cases (023, 030, 033, 034, 036, 037, 049) are expected to have Path-2 patterns; case 030 has ENTAILED claim → no contradiction → safe.
- Heldout ≥ 0.733: adding one correct case (020) only improves accuracy.

---

## Engine registration

Insert `RerankerFailureAnalyzer` in Layer 3 after `ClaimGroundingAnalyzer` (needs grounding evidence) and before `RetrievalDiagnosisAnalyzerV0` (so RERANKER_FAILURE from grounding evidence takes precedence over the catch-all diagnosis path).
