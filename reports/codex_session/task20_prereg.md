# Task 20 pre-registration — CHUNKING_BOUNDARY_ERROR over-firing

**Date:** 2026-06-18 · written BEFORE code.

## Root cause (established by instrumentation)

All 5 CLEAN→CHUNKING_BOUNDARY_ERROR probe false positives come from
`ParserValidationAnalyzer._text_only_fallback` → `_has_chunk_boundary_damage`
(`src/raggov/analyzers/parsing/parser_validation.py`). The rule flags any
**retrieval-order-consecutive** chunk pair where the left text ends in a dangling
function word (`of/for/to/in/on/with/and/or/the/a/an/is/are`) and the right text starts
lowercase — emitting `text_only_chunk_boundary_damage` / "Adjacent chunks split a sentence
across a boundary."

It never checks the two chunks are from the **same source document**. On multi-hop
retrieval (ALCE qampari / HotpotQA) every retrieved chunk is an independent passage from a
**distinct** document (verified: 10 chunks → 10 unique `doc_id`s per case). Passages that
happen to start mid-sentence (lowercase) are read as a within-document sentence split. A
genuine `CHUNKING_BOUNDARY_ERROR` is a sentence split **within one document's adjacent
chunks** — which requires the pair to share a source document.

This reads structured chunk provenance (`source_doc_id`), not query/passage text heuristics —
in-bounds for discipline rule #3.

## Change (one, narrow)

`_has_chunk_boundary_damage` additionally requires the consecutive pair to share the same
`source_doc_id` before flagging. Pass chunks (not just texts) so provenance is available.
Dangling-word + lowercase-start logic otherwise unchanged.

## Measured baseline (BEFORE — reproduced this session, post Tasks 18–19)

- Protected: 41/46 GREEN.
- Calib scored primary: 23/45 = 0.511.
- Probe overall: 47/145 = 0.324.
- Probe CLEAN→CHUNKING_BOUNDARY_ERROR false positives: **5** (of 30 CLEAN).

## Hard acceptance criteria

1. Protected baseline stays **41/46 GREEN**. *(revert trigger)*
2. Calib scored primary stays **≥ 23/45 = 0.511**. *(revert trigger)*
3. Full parsing/engine/decision-policy test suites stay green (no parser-validation TP
   regression). *(revert trigger)*
4. Probe overall accuracy does **not decrease** (≥ 47/145). *(revert trigger)*

**Success criterion:**
5. Probe CLEAN→CHUNKING_BOUNDARY_ERROR false positives drop materially: **5 → ≤1**.

Per the plan note, track the per-type FP count; another over-firer may inherit some freed
cases. If 1–4 hold but 5 lands at 2, keep only as a strict per-type improvement with zero TP
regression; else revert.
