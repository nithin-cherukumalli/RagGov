# Analyzer Predicate Analysis

## RetrievalAnomalyAnalyzer

- Current predicate: any score outlier, near-duplicate pair, or score cliff
  returns `warn/RETRIEVAL_ANOMALY`.
- Code location before edit: `src/raggov/analyzers/security/anomalies.py:48`.
- Separating signal: `retrieval_duplicate_chunks_11` has a near-duplicate pair
  with overlap `1.00`; Set A clean cases do not emit any retrieval anomaly.
- Predicate change justified: near-duplicate chunks are already direct retrieval
  anomaly evidence. Promoting duplicate-only retrieval anomaly evidence to
  `fail` does not depend on external providers or invented confidence scores.

## CitationFaithfulnessAnalyzerV0

- Current predicate: `unsupported_claim_ids` or `missing_citation_claim_ids`
  returns `warn/CITATION_MISMATCH`.
- Code location before edit: `src/raggov/analyzers/citation_faithfulness/analyzer.py:138`.
- Separating signal in production eval: Calib clean cases 011, 012, and 013
  include valid citations in `evals/govrag_calib/calib_150_seed.jsonl`, and
  `scripts/evaluate_govrag_calib.py` passes citation doc IDs into `RAGRun`.
  There are zero clean Calib or heldout cases with no citation objects.
- Real-failure signal: `citation_missing_26` and `quality_weak_grounding_39`
  have supported claims, supporting chunk evidence, one retrieved source, no
  citations, and no competing retrieval-noise or abstention evidence. The
  current analyzer labels this as `missing_citation_claim_ids`, then only warns.
- Predicate change justified: a supported/verifiable claim with no citation
  should be a blocking citation failure when citation absence is isolated. It
  should not override parser, retrieval, sufficiency, or abstention failures
  where missing citations are downstream/noisy. This remains a native
  heuristic/practical approximation, not calibrated confidence.

## RetrievalDiagnosisAnalyzerV0

- Current predicate: consumes `CitationFaithfulnessReport.missing_citation_claim_ids`
  and emits `warn/CITATION_MISMATCH`.
- Additional current predicate: consumes `RetrievalEvidenceProfile.noisy_chunk_ids`,
  computes `noise_ratio_warn` and `noise_min_chunks`, but still emits
  `warn/RETRIEVAL_ANOMALY` for any non-empty noisy chunk list below the fail
  threshold.
- Code locations inspected:
  - `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py:184`
  - `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py:317`
- Separating signal: `govrag-calib-seed-012` has one noisy chunk out of two.
  That meets `noise_ratio_warn=0.5` but fails `noise_min_chunks=2`; the analyzer
  records the minimum in limitations but does not enforce it for warn.
- Predicate change justified: below-warn-threshold noise should not produce a
  retrieval anomaly candidate. This uses the analyzer's existing explicit
  heuristic thresholds rather than adding a new threshold.

## ScopeViolationAnalyzer

- Current predicate: profile path emits `warn/SCOPE_VIOLATION` whenever any
  chunk has `query_relevance_label=IRRELEVANT`, and emits `fail` only when all
  retrieved chunks are irrelevant.
- Code location inspected: `src/raggov/analyzers/retrieval/scope.py:94`.
- Separating signal: `govrag-calib-seed-012` has one relevant answer-bearing
  chunk and one irrelevant supplemental chunk. Its label notes this is a
  clean/retrieval-noise ambiguity, but the expected primary remains `CLEAN`.
- Predicate change justified: a partial irrelevant tail should not be a scope
  warning when relevant chunks are present at least as strongly as irrelevant
  chunks. The scope warning remains for all-irrelevant retrieval and for cases
  where irrelevant chunks outnumber relevant chunks.

## Negative Finding

The handoff reproduction script, as written, does not carry Calib citation
objects into Set A runs. In that artificial shape, the citation signal does not
separate A and B: all five citation-relevant cases look like one missing
citation claim. The implementation target is therefore the production eval
surface, where citation fields are preserved.
