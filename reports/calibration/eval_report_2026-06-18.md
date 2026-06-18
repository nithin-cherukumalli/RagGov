# eval_report — 2026-06-18

**Method status:** heuristic_baseline / practical_approximation (uncalibrated).  
Scores EXACT primary-failure accuracy on available eval data — NOT a production-generalization guarantee.

Seeds used: [0]  
Modes: ['default', 'native']

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

### Induced Probe
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 145 | 80.0 | 0.5517 | 0.5517 | 0.5517 | 0.3696 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CLEAN | 30 | 13.0 | 0.4333 | None |
| INSUFFICIENT_CONTEXT | 30 | 4.0 | 0.1333 | None |
| UNSUPPORTED_CLAIM | 30 | 25.0 | 0.8333 | None |
| CITATION_MISMATCH | 30 | 29.0 | 0.9667 | None |
| CONTRADICTED_CLAIM | 15 | 0.0 | 0.0 | None |
| PROMPT_INJECTION | 10 | 9.0 | 0.9 | None |

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

### Induced Probe
| n | correct (mean) | accuracy mean | min | max | confidence_mean |
|----|----|----|----|----|-----|
| 145 | 82.0 | 0.5655 | 0.5655 | 0.5655 | 0.3696 |

**Per-type:**
| type | n | correct (mean) | accuracy mean | confidence_mean |
|------|---|----|----|------|
| CLEAN | 30 | 15.0 | 0.5 | None |
| INSUFFICIENT_CONTEXT | 30 | 4.0 | 0.1333 | None |
| UNSUPPORTED_CLAIM | 30 | 25.0 | 0.8333 | None |
| CITATION_MISMATCH | 30 | 29.0 | 0.9667 | None |
| CONTRADICTED_CLAIM | 15 | 0.0 | 0.0 | None |
| PROMPT_INJECTION | 10 | 9.0 | 0.9 | None |

## Spot-Case Parity (3 named cases vs. raggov_score.build_run)
| spot_label | case_id | expected | got | match | confidence |
|------|------|------|------|------|------|
| gc-001 | gc-001 | CLEAN | CLEAN | True | None |
| citation_probe | gc-PENDING | CITATION_MISMATCH | CITATION_MISMATCH | True | None |
| clean_probe | gc-PENDING | CLEAN | CLEAN | True | None |

> **Anchor note:** protected baseline check returned 42/46 (check_protected_baseline.py). The ledger anchor is 43/46 effective (including acceptable-alternative cases). Calib 23/45 confirmed. Probe 80/145 [default] / 82/145 [native] confirmed.