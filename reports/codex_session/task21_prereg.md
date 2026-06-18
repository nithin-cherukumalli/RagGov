# Task 21 pre-registration — NCV Jaccard duplicate-chunk over-firing

**Date:** 2026-06-18 · written BEFORE code.

## Root cause (established by instrumentation)

`NCVPipelineVerifier._check_context_assembly`
(`src/raggov/analyzers/verification/ncv.py`, ~line 591) emits a `FAIL` →
`INCONSISTENT_CHUNKS` when the best term-set Jaccard between any two retrieved chunks
exceeds **0.85**. On ALCE qampari, the same entity passage is retrieved from multiple
*distinct* documents (overlapping snippets), giving high lexical overlap that is **not** a
duplicate-chunk assembly defect.

Measured on the 3 CLEAN→INCONSISTENT_CHUNKS cases attributable to this path:
Jaccard = **0.898 / 0.854 / 0.961**, all `identical_text=False`, all different `source_doc_id`.
The genuine true positive (NCV `test_duplicate_chunks_fail_context_assembly`) is **identical
text → Jaccard 1.0**. So 0.85 is too loose: it conflates topical near-duplication with the
"same chunk assembled twice" defect the node is meant to catch.

## Change (one, narrow)

Raise the duplicate-chunk FAIL threshold from `0.85` to `0.97` (named constant
`_DUPLICATE_JACCARD_MIN`). A context-assembly duplicate must be essentially identical content
(tolerating whitespace/case via the existing term-set normalization); 0.85–0.96 topical
overlap no longer fails. Negation-pair branch and all other logic unchanged.

## Measured baseline (BEFORE — reproduced this session, post Tasks 18–20)

- Protected: 41/46 GREEN.
- Calib scored primary: 23/45 = 0.511.
- Probe overall: 54/145 = 0.372.
- Probe CLEAN→INCONSISTENT_CHUNKS via Jaccard duplicate path: **3** (Jaccard 0.85–0.96).

## Hard acceptance criteria

1. Protected baseline stays **41/46 GREEN**. *(revert trigger)*
2. Calib scored primary stays **≥ 23/45 = 0.511**. *(revert trigger)*
3. `test_duplicate_chunks_fail_context_assembly` (identical-text TP) stays green; full
   `tests/test_analyzers/test_ncv.py` green. *(revert trigger)*
4. Probe overall accuracy does **not decrease** (≥ 54/145). *(revert trigger)*

**Success criterion:**
5. Probe CLEAN→INCONSISTENT_CHUNKS Jaccard-path false positives drop: **3 → 0**.

The 2 remaining negation-path INCONSISTENT residuals are out of scope (irreducible without
NLI; documented in Task 19). If 1–4 hold but 5 lands at 1, keep as a strict improvement; else
revert.
