# eval_report Phase 2 — 2026-06-18

**Method status:** heuristic_baseline / practical_approximation (uncalibrated).  
Scores EXACT primary-failure accuracy — NOT a production-generalization guarantee.  
**Real heldout v1 (18/75 = 0.24) is now THE primary metric.**

Seeds: [0] | Modes: ['default', 'native']

## Mode: `default`
### Calib (train+dev+heldout)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 45 | 23.0 | 0.5111 | 0.5111 | 0.5111 | 0.1081 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CONTRADICTED_CLAIM | 11 | 6.0 | 0.5455 | None |
| CLEAN | 10 | 7.0 | 0.7 | None |
| INSUFFICIENT_CONTEXT | 5 | 1.0 | 0.2 | None |
| UNSUPPORTED_CLAIM | 4 | 3.0 | 0.75 | None |
| STALE_RETRIEVAL | 3 | 2.0 | 0.6667 | None |
| SCOPE_VIOLATION | 3 | 0.0 | 0.0 | None |
| CITATION_MISMATCH | 2 | 2.0 | 1.0 | None |
| PROMPT_INJECTION | 2 | 2.0 | 1.0 | None |
| RETRIEVAL_DEPTH_LIMIT | 2 | 0.0 | 0.0 | None |
| PRIVACY_VIOLATION | 1 | 0.0 | 0.0 | None |
| RETRIEVAL_ANOMALY | 1 | 0.0 | 0.0 | None |
| POST_RATIONALIZED_CITATION | 1 | 0.0 | 0.0 | None |

### Induced Probe (synthetic)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 0 | 0.0 | None | None | None | None |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|

### **Real Heldout v1** (production bar)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 75 | 18.0 | 0.24 | 0.24 | 0.24 | 0.5485 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CLEAN | 50 | 12.0 | 0.24 | None |
| CONTRADICTED_CLAIM | 25 | 6.0 | 0.24 | None |

**CLEAN-FP rate [heldout_real, default]:** 38/50 = 0.76 — breakdown: {'STALE_RETRIEVAL': 9, 'INSUFFICIENT_CONTEXT': 8, 'INCONSISTENT_CHUNKS': 7, 'UNSUPPORTED_CLAIM': 6, 'CONTRADICTED_CLAIM': 2, 'POST_RATIONALIZED_CITATION': 1, 'PROMPT_INJECTION': 1, 'PRIVACY_VIOLATION': 1, 'SCOPE_VIOLATION': 1, 'GENERATION_IGNORE': 1, 'RETRIEVAL_ANOMALY': 1}

## Mode: `native`
### Calib (train+dev+heldout)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 45 | 23.0 | 0.5111 | 0.5111 | 0.5111 | 0.1081 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CONTRADICTED_CLAIM | 11 | 6.0 | 0.5455 | None |
| CLEAN | 10 | 7.0 | 0.7 | None |
| INSUFFICIENT_CONTEXT | 5 | 1.0 | 0.2 | None |
| UNSUPPORTED_CLAIM | 4 | 3.0 | 0.75 | None |
| STALE_RETRIEVAL | 3 | 2.0 | 0.6667 | None |
| SCOPE_VIOLATION | 3 | 0.0 | 0.0 | None |
| CITATION_MISMATCH | 2 | 2.0 | 1.0 | None |
| PROMPT_INJECTION | 2 | 2.0 | 1.0 | None |
| RETRIEVAL_DEPTH_LIMIT | 2 | 0.0 | 0.0 | None |
| PRIVACY_VIOLATION | 1 | 0.0 | 0.0 | None |
| RETRIEVAL_ANOMALY | 1 | 0.0 | 0.0 | None |
| POST_RATIONALIZED_CITATION | 1 | 0.0 | 0.0 | None |

### Induced Probe (synthetic)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 0 | 0.0 | None | None | None | None |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|

### **Real Heldout v1** (production bar)
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 75 | 19.0 | 0.2533 | 0.2533 | 0.2533 | 0.5485 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CLEAN | 50 | 13.0 | 0.26 | None |
| CONTRADICTED_CLAIM | 25 | 6.0 | 0.24 | None |

**CLEAN-FP rate [heldout_real, native]:** 37/50 = 0.74 — breakdown: {'INSUFFICIENT_CONTEXT': 8, 'STALE_RETRIEVAL': 8, 'INCONSISTENT_CHUNKS': 8, 'UNSUPPORTED_CLAIM': 6, 'CONTRADICTED_CLAIM': 2, 'POST_RATIONALIZED_CITATION': 1, 'PROMPT_INJECTION': 1, 'PRIVACY_VIOLATION': 1, 'SCOPE_VIOLATION': 1, 'RETRIEVAL_ANOMALY': 1}

## NLI A/B Comparison (real heldout v1)
> ⚠️ Comparing native heuristic (conservative_ensemble fallback) against local_nli (cross-encoder/nli-deberta-v3-small).

| Policy | n | correct | accuracy | CLEAN-FP rate | CONTRADICTED recall |
|--------|---|---------|----------|---------------|---------------------|
| native (conservative_ensemble) | 75 | 18 | 0.24 | 0.76 | 0.24 |
| llm_entailment (→ heuristic fallback) | 75 | 6 | 0.08 | 0.88 | 0.0 |

**CLEAN-FP breakdown (native):** {'STALE_RETRIEVAL': 9, 'INSUFFICIENT_CONTEXT': 8, 'INCONSISTENT_CHUNKS': 7, 'UNSUPPORTED_CLAIM': 6, 'CONTRADICTED_CLAIM': 2, 'POST_RATIONALIZED_CITATION': 1, 'PROMPT_INJECTION': 1, 'PRIVACY_VIOLATION': 1, 'SCOPE_VIOLATION': 1, 'GENERATION_IGNORE': 1, 'RETRIEVAL_ANOMALY': 1}
**CLEAN-FP breakdown (entailment):** {'UNSUPPORTED_CLAIM': 21, 'INSUFFICIENT_CONTEXT': 9, 'STALE_RETRIEVAL': 7, 'INCONSISTENT_CHUNKS': 3, 'PROMPT_INJECTION': 1, 'PRIVACY_VIOLATION': 1, 'GENERATION_IGNORE': 1, 'RETRIEVAL_ANOMALY': 1}

## Spot-Case Parity (vs raggov_score.build_run)
| spot_label | case_id | expected | got | match | confidence |
|------|------|------|------|------|------|
| gc-001 | gc-001 | CLEAN | CLEAN | True | None |
| heldout_contra | heldout-real-ragtruth-13392 | CONTRADICTED_CLAIM | UNSUPPORTED_CLAIM | False | 1.2516 |
| heldout_clean | heldout-real-hotpotqa-5ab345db55429969a97a8122 | CLEAN | INSUFFICIENT_CONTEXT | False | 0.0 |

> **Anchors:** Calib 23/45. Probe 80/145 [default] / 82/145 [native]. Real heldout 18/75 = 0.24 [default]. Protected 43/46 effective.