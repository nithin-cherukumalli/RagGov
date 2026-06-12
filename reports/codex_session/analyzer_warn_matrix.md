# Analyzer Warn Matrix

Generated from the six-case reproduction script in the analyzer calibration
handoff, native mode.

| Case ID | Set | Primary | ParserValidationAnalyzer | ScopeViolationAnalyzer | CitationFaithfulnessAnalyzerV0 | TemporalSourceValidityAnalyzerV1 | RetrievalDiagnosisAnalyzerV0 | RetrievalAnomalyAnalyzer |
|---|---|---|---|---|---|---|---|---|
| govrag-calib-seed-011 | A_clean | CITATION_MISMATCH/GROUNDING | warn/METADATA_LOSS | - | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | - |
| govrag-calib-seed-012 | A_clean | SCOPE_VIOLATION/RETRIEVAL | warn/METADATA_LOSS | warn/SCOPE_VIOLATION | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | - |
| govrag-calib-seed-013 | A_clean | CITATION_MISMATCH/GROUNDING | warn/METADATA_LOSS | - | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | - |
| retrieval_duplicate_chunks_11 | B_real_failure | RETRIEVAL_ANOMALY/RETRIEVAL | warn/METADATA_LOSS | - | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | warn/RETRIEVAL_ANOMALY |
| citation_missing_26 | B_real_failure | CITATION_MISMATCH/GROUNDING | warn/METADATA_LOSS | - | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | - |
| quality_weak_grounding_39 | B_real_failure | CITATION_MISMATCH/GROUNDING | warn/METADATA_LOSS | - | warn/CITATION_MISMATCH | warn/STALE_RETRIEVAL | warn/CITATION_MISMATCH | - |

## Notes

- The reproduction script constructs Set A `RAGRun`s without passing
  `cited_doc_ids`, even though the Calib JSONL records for 011, 012, and 013
  contain valid citation objects. The production evaluator does pass citation
  IDs into `RAGRun`.
- `CitationFaithfulnessAnalyzerV0` and `RetrievalDiagnosisAnalyzerV0` emit the
  same `missing_citation_claim_ids=["claim_001"]` warning shape for all six
  reproduction cases.
- `RetrievalAnomalyAnalyzer` emits a Set-B-only signal for
  `retrieval_duplicate_chunks_11`: `near duplicate chunks c1 and c2 overlap=1.00`.
