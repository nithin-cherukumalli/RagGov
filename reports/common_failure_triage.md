# Common Failure Triage

Generated: `2026-06-12T04:53:55.742504+00:00`

## native

Pass rate: 41/46 (89.1%)
False CLEAN: 0
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 4 | 6 | 66.7% |
| citation | 5 | 5 | 100.0% |
| grounding | 6 | 7 | 85.7% |
| parser_chunking | 6 | 6 | 100.0% |
| retrieval | 5 | 6 | 83.3% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 5 | 5 | 100.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `grounding_date_hallucination_20` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_incomplete_38` | answer_quality | `UNSUPPORTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |

## external-enhanced

Pass rate: 41/46 (89.1%)
False CLEAN: 0
False SECURITY: 0
False INCOMPLETE: 0

### Category Pass Rates

| Category | Passed | Total | Pass Rate |
| :--- | ---: | ---: | ---: |
| answer_quality | 4 | 6 | 66.7% |
| citation | 5 | 5 | 100.0% |
| grounding | 6 | 7 | 85.7% |
| parser_chunking | 6 | 6 | 100.0% |
| retrieval | 5 | 6 | 83.3% |
| security | 5 | 6 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 5 | 5 | 100.0% |

### Failed Cases

| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalDiagnosisAnalyzerV0 | Retrieval evidence is generated but subtype mapping remains too coarse for this fixture. |
| `grounding_date_hallucination_20` | grounding | `UNSUPPORTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | none | ClaimGroundingAnalyzer (emitted expected evidence) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | RetrievalAnomalyAnalyzer (warn-level evidence not final) | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_incomplete_38` | answer_quality | `UNSUPPORTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` / `GENERATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | none | ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0 | Expected evidence exists but decision policy selected a different higher-ranked failure. |
