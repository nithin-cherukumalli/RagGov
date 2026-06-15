# Task 4 Pre-Registration — RETRIEVAL_DEPTH_LIMIT analyzer

**Date:** 2026-06-15  
**New file:** `src/raggov/analyzers/retrieval/depth.py`

---

## Dataset-independent predicate

Fire `RETRIEVAL_DEPTH_LIMIT` when ALL of the following hold:

1. `len(run.retrieved_chunks) <= k_floor` (configurable, default 5)
2. The query text contains a cardinal number N (digit or number-word: one…twelve, 13+)
3. `len(run.retrieved_chunks) < N`

If the query contains no cardinal number, the analyzer skips. This avoids any threshold tuned to the specific content of cases 004 or 019.

**Rationale for condition 2:**  
A query that explicitly states how many items it expects (e.g., "list all 5 requirements", "the six required attachments") provides a pipeline-agnostic signal that the retriever was asked to support N items. When the retrieved set is smaller than N, the retriever depth is demonstrably the limiting factor.

**k_floor semantics:**  
`k_floor` is the upper bound on retrieved-chunk count below which depth is considered suspiciously small. Default 5 means: only consider depth-limited when fewer than 6 chunks were returned. Callers can override via `config["k_floor"]`.

---

## Number extraction

Extract the first cardinal number found in the query via:
- Digit pattern: `\b(\d+)\b`
- Number-word pattern: `\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b` (case-insensitive)

When both are present, take the first match by position. Return `None` if no cardinal number is found.

---

## Cases 004 and 019

| Case | query | retrieved | implied N | fires? |
|------|-------|-----------|-----------|--------|
| 004  | "List all 5 requirements for the grant." | 2 | 5 | ✓ (2 < 5) |
| 019  | "List the six required grant attachments." | 2 | 6 | ✓ (2 < 6) |

---

## False-fire safety

- Cases with `len(retrieved_chunks) > k_floor`: skip on condition 1.
- Cases with `len(retrieved_chunks) = k_floor` and no cardinal number in query: skip on condition 2.
- Cases with `len(retrieved_chunks) = k_floor` and N ≤ retrieved: condition 3 is False.
- The engine position (Layer 3, after ClaimGrounding) means RETRIEVAL_DEPTH_LIMIT fires before RetrievalDiagnosisAnalyzerV0, giving it higher priority in the decision policy.

---

## Engine registration

Insert `RetrievalDepthLimitAnalyzer` in Layer 3, before `ScopeViolationAnalyzer`, so depth failures are surfaced before downstream citation/grounding aggregation analyzers.
