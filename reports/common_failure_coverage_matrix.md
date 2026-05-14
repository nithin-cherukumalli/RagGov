# RAG Failure Benchmark Coverage Matrix

**Date:** 2026-05-07
**Total Cases:** 46
**Passed:** 31
**Pass Rate:** 67.4%

## Performance by Category

| Category | Total | Passed | Pass Rate |
| :--- | :--- | :--- | :--- |
| answer_quality | 6 | 3 | 50.0% |
| citation | 5 | 3 | 60.0% |
| grounding | 7 | 3 | 42.9% |
| parser_chunking | 6 | 5 | 83.3% |
| retrieval | 6 | 4 | 66.7% |
| security | 6 | 5 | 83.3% |
| sufficiency | 5 | 5 | 100.0% |
| version_validity | 5 | 3 | 60.0% |

## Detailed Case Results

| Case ID | Category | Expected Primary | Actual Primary | Result |
| :--- | :--- | :--- | :--- | :--- |
| `parser_table_flattened_01` | parser_chunking | `TABLE_STRUCTURE_LOSS` | `TABLE_STRUCTURE_LOSS` | ✅ PASS |
| `parser_hierarchy_flattened_02` | parser_chunking | `HIERARCHY_FLATTENING` | `HIERARCHY_FLATTENING` | ✅ PASS |
| `parser_metadata_missing_03` | parser_chunking | `METADATA_LOSS` | `METADATA_LOSS` | ✅ PASS |
| `chunking_boundary_split_04` | parser_chunking | `CHUNKING_BOUNDARY_ERROR` | `CHUNKING_BOUNDARY_ERROR` | ✅ PASS |
| `summary_chunk_suppressed_table_05` | parser_chunking | `INSUFFICIENT_CONTEXT` | `UNSUPPORTED_CLAIM` | ❌ FAIL |
| `retrieval_miss_06` | retrieval | `INSUFFICIENT_CONTEXT` | `INSUFFICIENT_CONTEXT` | ✅ PASS |
| `retrieval_noise_07` | retrieval | `RETRIEVAL_ANOMALY` | `RETRIEVAL_ANOMALY` | ✅ PASS |
| `retrieval_top_k_too_small_08` | retrieval | `RETRIEVAL_DEPTH_LIMIT` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `retrieval_irrelevant_plausible_09` | retrieval | `SCOPE_VIOLATION` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `retrieval_unsupported_answer_10` | retrieval | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ✅ PASS |
| `retrieval_duplicate_chunks_11` | retrieval | `RETRIEVAL_ANOMALY` | `RETRIEVAL_ANOMALY` | ✅ PASS |
| `sufficiency_partial_12` | sufficiency | `INSUFFICIENT_CONTEXT` | `INSUFFICIENT_CONTEXT` | ✅ PASS |
| `sufficiency_missing_critical_13` | sufficiency | `INSUFFICIENT_CONTEXT` | `INSUFFICIENT_CONTEXT` | ✅ PASS |
| `sufficiency_missing_exception_14` | sufficiency | `INSUFFICIENT_CONTEXT` | `INSUFFICIENT_CONTEXT` | ✅ PASS |
| `sufficiency_missing_scope_15` | sufficiency | `INSUFFICIENT_CONTEXT` | `INSUFFICIENT_CONTEXT` | ✅ PASS |
| `sufficiency_stale_mistaken_16` | sufficiency | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ PASS |
| `grounding_unsupported_17` | grounding | `UNSUPPORTED_CLAIM` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `grounding_contradicted_18` | grounding | `CONTRADICTED_CLAIM` | `CONTRADICTED_CLAIM` | ✅ PASS |
| `grounding_numeric_hallucination_19` | grounding | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ✅ PASS |
| `grounding_date_hallucination_20` | grounding | `UNSUPPORTED_CLAIM` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `grounding_id_hallucination_21` | grounding | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ✅ PASS |
| `grounding_partial_support_22` | grounding | `UNSUPPORTED_CLAIM` | `CITATION_MISMATCH` | ❌ FAIL |
| `citation_phantom_23` | citation | `CITATION_MISMATCH` | `CITATION_MISMATCH` | ❌ FAIL |
| `citation_related_not_supporting_24` | citation | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ✅ PASS |
| `citation_contradicts_25` | citation | `CONTRADICTED_CLAIM` | `CONTRADICTED_CLAIM` | ✅ PASS |
| `citation_missing_26` | citation | `CITATION_MISMATCH` | `CITATION_MISMATCH` | ✅ PASS |
| `citation_post_rationalized_27` | citation | `POST_RATIONALIZED_CITATION` | `CITATION_MISMATCH` | ❌ FAIL |
| `version_expired_28` | version_validity | `STALE_RETRIEVAL` | `UNSUPPORTED_CLAIM` | ❌ FAIL |
| `version_superseded_29` | version_validity | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ PASS |
| `version_withdrawn_30` | version_validity | `STALE_RETRIEVAL` | `UNSUPPORTED_CLAIM` | ❌ FAIL |
| `version_not_yet_effective_31` | version_validity | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ PASS |
| `version_stale_not_cited_32` | version_validity | `STALE_RETRIEVAL` | `STALE_RETRIEVAL` | ✅ PASS |
| `security_prompt_injection_33` | security | `PROMPT_INJECTION` | `PROMPT_INJECTION` | ✅ PASS |
| `security_answer_steering_34` | security | `SUSPICIOUS_CHUNK` | `SUSPICIOUS_CHUNK` | ✅ PASS |
| `security_privacy_sensitive_35` | security | `PRIVACY_VIOLATION` | `PRIVACY_VIOLATION` | ✅ PASS |
| `security_retrieval_anomaly_only_36` | security | `RETRIEVAL_ANOMALY` | `UNSUPPORTED_CLAIM` | ❌ FAIL |
| `security_poisoning_explicit_37` | security | `SUSPICIOUS_CHUNK` | `SUSPICIOUS_CHUNK` | ✅ PASS |
| `quality_incomplete_38` | answer_quality | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ❌ FAIL |
| `quality_weak_grounding_39` | answer_quality | `CITATION_MISMATCH` | `CITATION_MISMATCH` | ✅ PASS |
| `quality_ambiguous_query_40` | answer_quality | `LOW_CONFIDENCE` | `LOW_CONFIDENCE` | ✅ PASS |
| `quality_ignores_context_41` | answer_quality | `CONTRADICTED_CLAIM` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `quality_overconfident_weak_evidence_42` | answer_quality | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | ✅ PASS |
| `parser_table_partial_loss_43` | parser_chunking | `TABLE_STRUCTURE_LOSS` | `TABLE_STRUCTURE_LOSS` | ✅ PASS |
| `retrieval_semantic_entropy_high_44` | answer_quality | `LOW_CONFIDENCE` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `grounding_complex_claim_split_45` | grounding | `UNSUPPORTED_CLAIM` | `INSUFFICIENT_CONTEXT` | ❌ FAIL |
| `security_poisoning_unlikely_anomaly_46` | security | `SUSPICIOUS_CHUNK` | `SUSPICIOUS_CHUNK` | ✅ PASS |