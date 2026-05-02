# Retrieval Diagnosis v0 Fixtures

## What these fixtures are

`retrieval_diagnosis_v0.jsonl` contains 12 regression cases for the v0
retrieval analysis layer (profiler + analyzers).  Each case records the
inputs given to `RetrievalEvidenceProfilerV0` and the four retrieval
analyzers, together with the exact outputs that the v0 heuristics produce.

## What these fixtures are NOT

**Fixtures validate current heuristic behavior only.**
They capture what the code produces today — not what a correct retrieval
judge would say.

**They are not calibration data.**
Calibration requires human-labeled cases with known ground truth, a
sufficient sample size (150–300+), and a controlled evaluation protocol.
These fixtures have none of those properties.

**They are not proof of semantic relevance or citation faithfulness.**
The v0 relevance signal is lexical overlap (term intersection), not semantic
similarity.  A chunk can overlap with query tokens and still be off-topic.
A chunk can have zero overlap and still be substantively relevant.
The same applies to citation and staleness checks.

**Future v1 requires human-labeled retrieval cases.**
A production-grade evaluation layer needs:
- Human annotators who read the query, chunk, and answer
- Labeled relevance judgments (relevant / partial / irrelevant)
- Labeled citation faithfulness (supported / unsupported / phantom)
- Labeled staleness (valid / superseded / outdated by policy change)
- At least one full RAG pipeline run per case to test end-to-end behavior

## Case inventory

| case_id | Signal tested | Expected scope | Expected citation | Expected stale | Expected inconsistency |
|---|---|---|---|---|---|
| case_01_all_chunks_relevant | Relevance — all relevant | pass | pass | pass | pass |
| case_02_one_irrelevant_chunk | Relevance — partial | warn | pass | pass | pass |
| case_03_all_chunks_irrelevant | Relevance — all irrelevant | fail | pass | pass | pass |
| case_04_phantom_citation | Citation — phantom | pass | fail | pass | pass |
| case_05_valid_citation | Citation — all valid | pass | pass | pass | pass |
| case_06_stale_document_by_age | Staleness — stale | pass | pass | fail | pass |
| case_07_fresh_document | Staleness — fresh | pass | pass | pass | pass |
| case_08_negation_contradiction_candidate | Inconsistency — negation pair | pass | pass | pass | warn |
| case_09_no_contradiction | Inconsistency — none | pass | pass | pass | pass |
| case_10_no_retrieved_chunks | Edge — empty retrieval | skip | pass* | skip | skip |
| case_11_cited_doc_ids_missing | Edge — no cited IDs | pass | pass* | pass | pass |
| case_12_corpus_metadata_missing | Edge — no corpus | pass | pass | pass* | pass |

\* Profile-path behavioral difference from legacy: when a `RetrievalEvidenceProfile`
is pre-attached to the run, `CitationMismatchAnalyzer` passes (no phantoms detected)
rather than skipping.  `StaleRetrievalAnalyzer` passes (no stale docs detected)
rather than skipping for missing corpus metadata.

## Known limitations of v0 fixtures

- **Staleness drift**: cases 06 and 07 use hardcoded timestamps.
  Case 07 (fresh document) uses `2026-04-15` which will become stale
  (>180 days) around October 2026.  Update the timestamp when that occurs.
- **Lexical relevance is not semantic relevance**: cases 01–03 test the
  term-overlap heuristic only.  They do not test whether the retrieved
  content is actually useful for answering the query.
- **Negation heuristic is approximate**: case 08 uses the exact token-window
  negation pattern from `InconsistentChunksAnalyzer`.  Real contradictions
  can appear without negation words; negation words can appear without
  contradiction.
- **Single-signal cases**: each case is designed to exercise one signal in
  isolation.  Real RAG runs exhibit multiple simultaneous failures.

## Running the evaluation harness

```bash
# Print console summary
python scripts/evaluate_retrieval_analyzers.py

# Print summary + write JSON report
python scripts/evaluate_retrieval_analyzers.py --output reports/retrieval_v0_eval.json

# Use a different fixture file
python scripts/evaluate_retrieval_analyzers.py --fixture path/to/custom.jsonl
```

## Updating fixtures

When the v0 heuristics are intentionally changed (new threshold, new signal,
bug fix), update the `expected_profile` and `expected_analyzer_results` fields
to match the new behavior.  Always document the change in the PR description
and note whether the change improves or degrades agreement with any existing
human judgments.
