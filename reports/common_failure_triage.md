# Common Failure Triage

Generated: `2026-05-13T08:50:19.245988+00:00`

## native

Pass rate: 30/46 (65.2%)
False CLEAN: 1
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 3 | 6 | 50.0% |
| citation | 3 | 5 | 60.0% |
| grounding | 3 | 7 | 42.9% |
| parser_chunking | 5 | 6 | 83.3% |
| retrieval | 4 | 6 | 66.7% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 2 | 5 | 40.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `summary_chunk_suppressed_table_05` | parser_chunking | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ParserValidationAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `retrieval_irrelevant_plausible_09` | retrieval | `SCOPE_VIOLATION` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ScopeViolationAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_unsupported_17` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `grounding_date_hallucination_20` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_partial_support_22` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `CITATION_MISMATCH` / `GROUNDING` | none | ClaimGroundingAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `citation_phantom_23` | citation | `CITATION_MISMATCH` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessAnalyzerV0 (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_post_rationalized_27` | citation | `POST_RATIONALIZED_CITATION` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessProbe (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_expired_28` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_withdrawn_30` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_stale_not_cited_32` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `CLEAN` / `UNKNOWN` | false_clean | TemporalSourceValidityAnalyzerV1 (warn-level evidence not final) | Expected failure evidence was absent or only advisory/warn-level, so clean was not blocked. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_incomplete_38` | answer_quality | `UNSUPPORTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `retrieval_semantic_entropy_high_44` | answer_quality | `LOW_CONFIDENCE` / `CONFIDENCE` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | SemanticEntropyAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_complex_claim_split_45` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |

## external-enhanced

Pass rate: 31/46 (67.4%)
False CLEAN: 0
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 3 | 6 | 50.0% |
| citation | 3 | 5 | 60.0% |
| grounding | 3 | 7 | 42.9% |
| parser_chunking | 5 | 6 | 83.3% |
| retrieval | 4 | 6 | 66.7% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 3 | 5 | 60.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `summary_chunk_suppressed_table_05` | parser_chunking | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ParserValidationAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `retrieval_irrelevant_plausible_09` | retrieval | `SCOPE_VIOLATION` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ScopeViolationAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_unsupported_17` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `grounding_date_hallucination_20` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_partial_support_22` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `CITATION_MISMATCH` / `GROUNDING` | none | ClaimGroundingAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `citation_phantom_23` | citation | `CITATION_MISMATCH` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessAnalyzerV0 (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_post_rationalized_27` | citation | `POST_RATIONALIZED_CITATION` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessProbe (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_expired_28` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_withdrawn_30` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_incomplete_38` | answer_quality | `UNSUPPORTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `retrieval_semantic_entropy_high_44` | answer_quality | `LOW_CONFIDENCE` / `CONFIDENCE` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | SemanticEntropyAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_complex_claim_split_45` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
