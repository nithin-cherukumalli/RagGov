# CLEAN false-positive triage — the native-precision pivot target list

**Date:** 2026-06-22. Native mode, locked real heldout (75 rows). This is the foundation for the
post-Phase-2 pivot: drive CLEAN-FP (0.76) down by fixing the native analyzers that over-fire,
NOT by adding an entailment tier (see `phase2_entailment_tier_NOGO.md`).

## The 38 CLEAN false-positives, by firing analyzer
| n | failure type | winning analyzer(s) | signal | heldout TPs | Calib gold/caught |
|---|--------------|---------------------|--------|-------------|-------------------|
| 8 | INCONSISTENT_CHUNKS | InconsistentChunksAnalyzer | `[profile] cX <-> cY` | **0** | **0 / 0** |
| 8 | INSUFFICIENT_CONTEXT | SufficiencyAnalyzer | `missing_scope_condition`, `missing_exception` | 0 | 5 / 1 |
| 8 | STALE_RETRIEVAL | RetrievalDiagnosisAnalyzerV0 (6), TemporalSourceValidityAnalyzerV1 (2) | `version_retrieval_failure` | 0 | 3 / 2 |
| 6 | UNSUPPORTED_CLAIM | ClaimGroundingAnalyzer | `diagnostic_rollup_heuristic_v0` (noisy_retrieval/value_error) | 0 | — |
| 2 | CONTRADICTED_CLAIM | ClaimGroundingAnalyzer | rollup value_error | (mismapped) | — |
| 1 | POST_RATIONALIZED_CITATION | CitationFaithfulnessProbe | stale_docs rollup | — | — |
| 1 | PROMPT_INJECTION | PromptInjectionAnalyzer | `[profile] c0<->c4` | — | — |
| 1 | PRIVACY_VIOLATION | PrivacyAnalyzer | stale_docs rollup | — | — |
| 1 | SCOPE_VIOLATION | ScopeViolationAnalyzer | off_topic_retrieval | — | — |
| 1 | RETRIEVAL_ANOMALY | RetrievalDiagnosisAnalyzerV0 | retrieval_noise | — | — |

## Two reframing facts
1. **The heldout can only measure precision right now.** All 25 gold-FAIL rows carry the mismapped
   CONTRADICTED_CLAIM label (RAGTruth span → whole-answer; 4 judges agree). So there are NO real
   gold INCONSISTENT/INSUFFICIENT/STALE rows. These analyzers fire on the heldout ONLY as false
   positives — tightening them costs zero heldout recall. The guard rails are Calib + protected-46.
2. **INCONSISTENT_CHUNKS is pure false-positive** on every dataset we can measure: 0 TPs on the
   heldout AND 0 gold/0 caught on Calib, but 8 FPs on the heldout. On this data the analyzer
   contributes only noise.

## Recommended first increment (pre-register before code)
**Target: InconsistentChunksAnalyzer `[profile]` path → reclaim up to 8 CLEAN-FP, ~0 recall risk.**
- Pre-reg `clean_fp_task1_prereg.md`: hypothesis = the `[profile] cX<->cY` chunk-pair signal
  over-fires on faithful answers; gate it (raise the bar / make it advisory unless a polarity-
  opposed proposition is present — the Task 19 mechanism, evidently still too loose on real data).
- Hard criteria (revert on any fail): protected 43/46 unchanged; Calib 23/45 unchanged; real
  heldout native CLEAN-FP DOWN (target ≥ +6 of the 8); no NEW failure-type FP introduced; named
  TPs intact. MUST first confirm InconsistentChunksAnalyzer is not load-bearing for any
  protected-46 TP — if it is, scope the fix to the FP sub-signal only (Task 24/26 lesson:
  FPs often share the TP mechanism; revert beats a false fix).

## Then, in priority order
2. **SufficiencyAnalyzer scope-condition (8).** `missing_scope_condition`/`missing_exception`
   penalize correctly-scoped context. Has Calib TPs (5 gold/1 caught) — scope the fix to the
   scope-condition sub-rule, not the whole analyzer.
3. **STALE_RETRIEVAL `version_retrieval_failure` (8).** Two analyzers; has Calib TPs (3/2) — careful.
4. **The 5 security/citation singletons** (PROMPT_INJECTION, PRIVACY_VIOLATION, SCOPE_VIOLATION,
   POST_RATIONALIZED_CITATION, RETRIEVAL_ANOMALY on CLEAN answers). Low count but high-severity FPs
   (security false alarms on faithful answers) — worth a look once the big buckets land.

## Discipline
One analyzer per increment, pre-registered, measured on the locked heldout + Calib + protected-46,
reverted on any regression. The heldout measures precision (CLEAN-FP) only until the CONTRADICTED
labels are adjudicated; do not chase recall against the mismapped rows.
