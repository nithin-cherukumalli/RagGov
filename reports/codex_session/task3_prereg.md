# Task 3 Pre-Registration — CITATION_MISMATCH vs UNSUPPORTED_CLAIM routing

**Date:** 2026-06-15  
**Decision:** Option **(a)** — fix the predicate in `CitationFaithfulnessAnalyzerV0`.

---

## Cases under review

| Case | Expected | Answer citation | Claim status |
|------|----------|-----------------|--------------|
| 025  | `UNSUPPORTED_CLAIM` | none | claim mixes docs, genuinely unsupported |
| 029  | `CITATION_MISMATCH` | `[travel-budget]` (retrieved, wrong doc) | claim entailed by c1, not by cited doc |
| 030  | `CITATION_MISMATCH` | `[msa-retention]` (retrieved, wrong doc) | claim entailed by c1, not by cited doc |

Case 025's fixture expects `UNSUPPORTED_CLAIM`; the task description that said otherwise was imprecise. The fixture is authoritative. No change needed for case 025.

---

## Root-cause trace for cases 029 and 030

1. `ClaimGroundingAnalyzer` — the claim IS entailed by the top retrieved chunk (c1), so it returns PASS.
2. `evidence_layer.py` — `_CITATION_REGEX = r"\[(?:c|doc\s*)(\d+)\]"` only matches numeric IDs like `[c1]`/`[doc1]`. The doc-name bracket references `[travel-budget]` and `[msa-retention]` are not matched, so `cited_ids = []` in each `ClaimEvidenceRecord`.
3. `CitationFaithfulnessAnalyzerV0._citation_sources()` — falls back to the `LEGACY_CITATION_IDS` path using `run.cited_doc_ids` (populated correctly from the fixture's `citations` field by `_record_to_run()`).
4. `_support_label()` — cited doc IS in the retrieved set (not phantom), but does NOT support the claim (supporting chunk is from a different doc). Hits the `"retrieved_uncited_chunk"` branch → returns `CitationSupportLabel.UNSUPPORTED`.
5. `report.unsupported_claim_ids` is populated.
6. Fail-level CITATION_MISMATCH branch (line 127):
   ```python
   if report.unsupported_claim_ids
       and self._answer_has_explicit_citation_marker(run)
       and self._answer_has_specific_value(run):
   ```
   - Case 029: `"6110"` matches `\b\d{2,4}\b` → fires CITATION_MISMATCH ✓ (already correct)
   - Case 030: `"five years"` has no digits → gate fails → falls to **warn**-level only → engine returns CLEAN

---

## Selected fix: option (a)

**Predicate change (dataset-independent terms):**

Replace the `_answer_has_specific_value` gate with `_cited_doc_is_retrieved`:

> Fire fail-level CITATION_MISMATCH when:
> 1. `report.unsupported_claim_ids` is non-empty, AND
> 2. The answer contains an explicit bracket citation marker (`[` and `]`), AND
> 3. At least one `run.cited_doc_ids` entry is present in the retrieved chunk set.

Condition 3 ("cited doc is retrieved") captures the structural property that the citation is *real but wrong* — the cited document exists in the retrieval result but does not support the claim. This is dataset-independent: it does not inspect the content of the answer, domain vocabulary, or numeric patterns.

**Method:**
```python
def _cited_doc_is_retrieved(self, run: RAGRun) -> bool:
    retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
    return bool(set(run.cited_doc_ids) & retrieved_doc_ids)
```

Replace line 127:
```python
# Before
if report.unsupported_claim_ids and self._answer_has_explicit_citation_marker(run) and self._answer_has_specific_value(run):
# After
if report.unsupported_claim_ids and self._answer_has_explicit_citation_marker(run) and self._cited_doc_is_retrieved(run):
```

---

## Safety analysis

| Case | `cited_doc_ids` | cited doc in retrieved? | `_answer_has_explicit_citation_marker` | Expected result | After fix |
|------|----------------|------------------------|---------------------------------------|----------------|-----------|
| 025  | `[]`           | N/A (empty)             | False (no brackets in answer)          | UNSUPPORTED_CLAIM | unchanged — neither gate fires |
| 029  | `["travel-budget"]` | Yes | True | CITATION_MISMATCH | same (was already firing) |
| 030  | `["msa-retention"]` | Yes | True | CITATION_MISMATCH | **now fires** (gate changed) |

The fix does not affect any case where `run.cited_doc_ids = []` or where the answer has no bracket notation.

---

## Acceptance criteria mapping

- ≥ 2 of {025, 029, 030} produce expected outputs: 025 → UNSUPPORTED_CLAIM ✓, 029 → CITATION_MISMATCH ✓, 030 → CITATION_MISMATCH ✓ (3 of 3)
- Protected pin cases (023, 030, 033, 034, 036, 037, 049): case 030 is in the protected set as "must not become CLEAN" — after fix it becomes CITATION_MISMATCH (fail), so the pin is satisfied.
- Heldout ≥ 0.733: fixing case 030 from CLEAN to correct CITATION_MISMATCH only improves accuracy.
