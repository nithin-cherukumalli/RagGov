# Common Failure Triage

Generated: `2026-05-14T11:11:44.570996+00:00`

## native

Pass rate: 31/46 (67.4%)
False CLEAN: 0
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 4 | 6 | 66.7% |
| citation | 2 | 5 | 40.0% |
| grounding | 4 | 7 | 57.1% |
| parser_chunking | 5 | 6 | 83.3% |
| retrieval | 3 | 6 | 50.0% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 3 | 5 | 60.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `summary_chunk_suppressed_table_05` | parser_chunking | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ParserValidationAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `retrieval_irrelevant_plausible_09` | retrieval | `SCOPE_VIOLATION` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ScopeViolationAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `retrieval_unsupported_answer_10` | retrieval | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_numeric_hallucination_19` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_id_hallucination_21` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_partial_support_22` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_phantom_23` | citation | `CITATION_MISMATCH` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessAnalyzerV0 (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_related_not_supporting_24` | citation | `UNSUPPORTED_CLAIM` / `GROUNDING` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_post_rationalized_27` | citation | `POST_RATIONALIZED_CITATION` / `GROUNDING` | `CITATION_MISMATCH` / `GROUNDING` | none | CitationFaithfulnessProbe (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_expired_28` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_withdrawn_30` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `CONTRADICTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_overconfident_weak_evidence_42` | answer_quality | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |

## external-enhanced

Pass rate: 31/46 (67.4%)
False CLEAN: 0
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 4 | 6 | 66.7% |
| citation | 2 | 5 | 40.0% |
| grounding | 4 | 7 | 57.1% |
| parser_chunking | 5 | 6 | 83.3% |
| retrieval | 3 | 6 | 50.0% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 3 | 5 | 60.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `summary_chunk_suppressed_table_05` | parser_chunking | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ParserValidationAnalyzer | Expected analyzer did not emit the gold failure type for the available fixture evidence. |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `retrieval_irrelevant_plausible_09` | retrieval | `SCOPE_VIOLATION` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | none | ScopeViolationAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `retrieval_unsupported_answer_10` | retrieval | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_numeric_hallucination_19` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_id_hallucination_21` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `grounding_partial_support_22` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_phantom_23` | citation | `CITATION_MISMATCH` / `GROUNDING` | `CITATION_MISMATCH` / `RETRIEVAL` | none | CitationFaithfulnessAnalyzerV0 (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_related_not_supporting_24` | citation | `UNSUPPORTED_CLAIM` / `GROUNDING` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `citation_post_rationalized_27` | citation | `POST_RATIONALIZED_CITATION` / `GROUNDING` | `CITATION_MISMATCH` / `GROUNDING` | none | CitationFaithfulnessProbe (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_expired_28` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `version_withdrawn_30` | version_validity | `STALE_RETRIEVAL` / `RETRIEVAL` | `POST_RATIONALIZED_CITATION` / `GROUNDING` | none | TemporalSourceValidityAnalyzerV1 (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `CONTRADICTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_overconfident_weak_evidence_42` | answer_quality | `UNSUPPORTED_CLAIM` / `GROUNDING` | `UNSUPPORTED_CLAIM` / `GENERATION` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
