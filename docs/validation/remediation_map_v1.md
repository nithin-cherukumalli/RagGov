# RagGov Analyzer Remediation Map v1

**Created**: 2026-04-23
**Based on**: `baseline_validation_v1.json` (10 cases)
**Purpose**: Prioritize analyzer improvements based on measured performance

---

## Executive Summary

**Calibration Status**: PROVISIONAL (10 samples - insufficient for stable thresholds)

**Coverage**: 10/48 golden set items have fixtures (38-item gap)

**Overall Match Rate**: 90% (9/10 cases)

**Critical Finding**: 12 analyzers have <60% accuracy and require immediate review before any threshold tuning.

---

## Analyzer Classification Summary

| Calibration Status | Count | Analyzers |
|--------------------|-------|-----------|
| **DETERMINISTIC** (no calibration needed) | 4 | PromptInjectionAnalyzer, PrivacyAnalyzer, NCVPipelineVerifier, CitationMismatchAnalyzer |
| **PROVISIONAL** (10-50 samples) | 3 | ClaimGroundingAnalyzer, CitationFaithfulnessProbe, SufficiencyAnalyzer |
| **NOT_CALIBRATED** (<30 samples or unstable) | 7 | ParserValidationAnalyzer, StaleRetrievalAnalyzer, ScopeViolationAnalyzer, PoisoningHeuristicAnalyzer, RetrievalAnomalyAnalyzer, SemanticEntropyAnalyzer, + 1 more |

---

## Priority 1: Unstable Analyzers Requiring Immediate Attention

These analyzers have <60% accuracy and should NOT be used for threshold calibration until fixed.

### 1. SemanticEntropyAnalyzer - 10% accuracy ⚠️ CRITICAL

**Current Issues**:
- Misnomed: Computes claim-label entropy, NOT semantic entropy per Farquhar et al.
- 10% accuracy indicates fundamental implementation problem
- Threshold (`entropy_threshold=1.2`) is arbitrary

**Recommended Actions**:
1. **Rename** to `ClaimLabelEntropyAnalyzer` (Phase 2C task)
2. Update docstring to clarify NOT semantic entropy
3. Investigate why accuracy is so low - may need algorithm fix
4. Consider disabling until proper semantic entropy implemented

**Sample Count Needed**: 150+ (if keeping as proxy metric)
**Effort**: HIGH - requires rename + algorithm investigation
**Priority**: IMMEDIATE (misnaming + low accuracy)

---

### 2. ParserValidationAnalyzer - 30% accuracy ⚠️ HIGH

**Current Issues**:
- 30% accuracy suggests heuristic patterns miss many failures
- Threshold provenance: HEURISTIC_DEFAULT (not validated)
- May be too specific to certain parser failure modes

**Recommended Actions**:
1. Manual review of false negatives - which parser failures are missed?
2. Expand pattern library for table/hierarchy detection
3. Consider structure-aware validation (not just text patterns)

**Sample Count Needed**: 50+ with parser-specific failures
**Effort**: MEDIUM - pattern expansion + validation
**Priority**: HIGH (critical for PARSING stage attribution)

---

### 3. Sufficiency Analyzer - 20% accuracy ⚠️ HIGH

**Current Issues**:
- 20% accuracy indicates token count + keyword heuristics insufficient
- Current approach too simplistic for context sufficiency
- Threshold provenance: HEURISTIC_DEFAULT

**Recommended Actions**:
1. Review false positives - is it over-flagging sufficient contexts?
2. Consider semantic similarity to question (not just keyword match)
3. Add minimum evidence span requirement

**Sample Count Needed**: 50+ with sufficiency-specific test cases
**Effort**: MEDIUM - algorithm improvement
**Priority**: HIGH (affects should_have_answered decisions)

---

### 4. ScopeViolationAnalyzer - 20% accuracy

**Current Issues**:
- Keyword overlap insufficient for scope determination
- No clear baseline for what constitutes "in scope"
- Threshold provenance: HEURISTIC_DEFAULT

**Recommended Actions**:
1. Define explicit scope boundaries per corpus
2. Use embedding-based scope distance (not just keyword)
3. Add corpus-specific scope definitions

**Sample Count Needed**: 30+ with clear scope violations
**Effort**: MEDIUM
**Priority**: MEDIUM (useful for governance but not critical path)

---

### 5. Stale Retrieval Analyzer - 20% accuracy

**Current Issues**:
- Date extraction may be failing
- `max_age_days=180` is arbitrary, not tuned to use case
- May be flagging documents that are legitimately still valid

**Recommended Actions**:
1. Improve date extraction (add more date formats)
2. Make `max_age_days` configurable per corpus
3. Add severity levels (Phase 2D task)

**Sample Count Needed**: 20+ with stale documents
**Effort**: LOW - mostly configuration + date parsing
**Priority**: LOW (demoted to Phase 2D)

---

### 6. Poisoning Heuristic Analyzer - 20% accuracy

**Current Issues**:
- No research alignment (arbitrary heuristics)
- High false positive risk
- Unclear what constitutes "poisoning" vs normal content variation

**Recommended Actions**:
1. Research corpus poisoning detection methods
2. Align heuristics to published poisoning taxonomies
3. OR disable until proper detection method implemented

**Sample Count Needed**: 50+ with known poisoned examples
**Effort**: HIGH - research + algorithm redesign
**Priority**: MEDIUM (security important, but low sample representation)

---

### 7. Retrieval Anomaly Analyzer - 10% accuracy

**Current Issues**:
- 10% accuracy suggests approach is not working
- Statistical outlier detection may be too simplistic
- Threshold provenance: ARBITRARY

**Recommended Actions**:
1. Review what anomalies are being detected vs expected
2. Consider distribution-based anomaly detection (not fixed thresholds)
3. Align with retrieval research (MMR diversity, rank instability patterns)

**Sample Count Needed**: 30+ with retrieval anomalies
**Effort**: MEDIUM - algorithm improvement
**Priority**: MEDIUM (useful for debugging but not critical)

---

## Priority 2: Provisional Analyzers (Need More Samples for Calibration)

These analyzers have 50%+ accuracy but insufficient samples for stable thresholds.

### 8. ClaimGroundingAnalyzer - 50% accuracy

**Current Status**:
- PROVISIONAL calibration (10 samples)
- Partially research-aligned (NLI-based)
- Threshold provenance: HEURISTIC_DEFAULT

**Recommended Actions**:
1. Collect 150+ grounding-specific test cases
2. Calibrate NLI confidence thresholds
3. Validate against ARES calibration set

**Sample Count Needed**: 150+ for CALIBRATED status
**Effort**: MEDIUM (sample collection + calibration)
**Priority**: HIGH (core grounding analyzer)

---

### 9. Citation Faithfulness Probe - 50% accuracy

**Current Status**:
- PROVISIONAL calibration (10 samples)
- Heuristic (anchor/predicate overlap)
- **MISNOMED**: checks correctness, NOT faithfulness

**Recommended Actions**:
1. Rename to `CitationCorrectnessProbe` (Phase 2B task)
2. Split failure types: CITATION_MISMATCH, CITATION_PARTIAL_SUPPORT, CITATION_UNSUPPORTED
3. Document true faithfulness as future work requiring perturbation

**Sample Count Needed**: 50+ with citation issues
**Effort**: LOW - rename + doc updates
**Priority**: IMMEDIATE (misnaming fix required)

---

## Priority 3: Stable Deterministic Analyzers (No Calibration Needed)

These analyzers are rule-based and don't require threshold calibration.

### 10. Prompt Injection Analyzer - 20% accuracy (but DETERMINISTIC)

**Note**: Low "accuracy" in confusion matrix may be misleading - this is a deterministic check that fires only on specific patterns. The 20% may reflect limited test coverage of injection attempts.

**Recommended Actions**:
1. Verify this analyzer is working correctly on known injection patterns
2. Expand test coverage with more injection variants
3. No threshold tuning needed (pattern-based)

**Effort**: LOW
**Priority**: LOW (verify functionality, not tune)

---

### 11. Privacy Analyzer, Citation Mismatch Analyzer, NCV Pipeline Verifier

**Status**: DETERMINISTIC (no calibration needed)

**Recommended Actions**: None (functioning as designed)

---

## Stage-Aware Remediation Recommendations

Based on baseline validation, prioritize remediation by pipeline stage:

| Stage | Unstable Analyzers | Priority | Recommended Fix |
|-------|-------------------|----------|-----------------|
| **PARSING** | ParserValidationAnalyzer (30%) | HIGH | Expand pattern library, add structure validation |
| **RETRIEVAL** | StaleRetrievalAnalyzer (20%), ScopeViolationAnalyzer (20%), RetrievalAnomalyAnalyzer (10%) | MEDIUM | Improve heuristics, add semantic detection |
| **GROUNDING** | ClaimGroundingAnalyzer (50%), CitationFaithfulnessProbe (50%) | HIGH | Rename, collect more samples, calibrate |
| **SUFFICIENCY** | SufficiencyAnalyzer (20%) | HIGH | Improve algorithm, add semantic similarity |
| **SECURITY** | PoisoningHeuristicAnalyzer (20%) | MEDIUM | Research-align or disable |
| **CONFIDENCE** | SemanticEntropyAnalyzer (10%) | CRITICAL | Rename, fix algorithm or disable |

---

## Sample Collection Priorities

To move from PROVISIONAL to CALIBRATED status, collect samples in this priority order:

1. **Grounding failures** (50 → 150 samples)
   - Needed for: ClaimGroundingAnalyzer, CitationCorrectnessProbe
   - High leverage: Core failure detection
   - Effort: MEDIUM (manual grounding annotation required)

2. **Parser failures** (10 → 50 samples)
   - Needed for: ParserValidationAnalyzer
   - High leverage: Catches upstream failures
   - Effort: LOW (parser output comparison straightforward)

3. **Sufficiency failures** (10 → 50 samples)
   - Needed for: SufficiencyAnalyzer
   - High leverage: Affects abstention decisions
   - Effort: MEDIUM (subjective judgment of "enough context")

4. **Security failures** (10 → 30 samples)
   - Needed for: PoisoningHeuristicAnalyzer
   - Medium leverage: Important but rare
   - Effort: HIGH (requires crafting poisoned examples)

---

## Known False Positive Patterns

Based on manual inspection and baseline results:

1. **Citation Faithfulness Probe**: Flags citations when chunk boundaries split evidence across multiple chunks
2. **Sufficiency Analyzer**: Over-flags short but complete answers (keyword heuristic too strict)
3. **Scope Violation Analyzer**: Flags legitimate cross-domain queries that span policy areas
4. **Semantic Entropy Analyzer**: Unknown (10% accuracy suggests systematic failure)

---

## Next Steps for Phase 1B Completion

1. **IMMEDIATE**:
   - [ ] Rename `SemanticEntropyAnalyzer` to `ClaimLabelEntropyAnalyzer` (Phase 2C)
   - [ ] Rename `CitationFaithfulnessProbe` to `CitationCorrectnessProbe` (Phase 2B)
   - [ ] Investigate SemanticEntropyAnalyzer's 10% accuracy (critical bug?)

2. **SHORT TERM** (before Phase 2A):
   - [ ] Expand parser validation patterns (target: 50% → 70% accuracy)
   - [ ] Improve sufficiency algorithm (target: 20% → 60% accuracy)
   - [ ] Create 40 additional grounding test cases (10 → 50 samples)

3. **LONG TERM** (parallel with Phase 2):
   - [ ] Collect remaining 38 golden set fixtures
   - [ ] Expand to 150+ samples for CALIBRATED status
   - [ ] Re-run baseline validation after Phase 2C analyzer renames

---

## Calibration Readiness Summary

| Analyzer | Current Status | Samples Needed | Accuracy | Ready for Calibration? |
|----------|---------------|----------------|----------|------------------------|
| ClaimGroundingAnalyzer | PROVISIONAL | 140 more | 50% | After algorithm fix |
| CitationFaithfulnessProbe | PROVISIONAL | 40 more | 50% | After rename to CitationCorrectnessProbe |
| SufficiencyAnalyzer | NOT_CALIBRATED | 40 more | 20% | NO - algorithm fix first |
| ParserValidationAnalyzer | NOT_CALIBRATED | 40 more | 30% | NO - pattern expansion first |
| StaleRetrievalAnalyzer | NOT_CALIBRATED | 10 more | 20% | MAYBE - simple config fix |
| ScopeViolationAnalyzer | NOT_CALIBRATED | 20 more | 20% | NO - algorithm fix first |
| PoisoningHeuristicAnalyzer | NOT_CALIBRATED | 40 more | 20% | NO - research alignment first |
| RetrievalAnomalyAnalyzer | NOT_CALIBRATED | 20 more | 10% | NO - algorithm redesign first |
| SemanticEntropyAnalyzer | NOT_CALIBRATED | N/A | 10% | NO - rename + fix first |

**Bottom Line**: Only deterministic analyzers are currently reliable. All threshold-based analyzers require fixes before calibration.

---

**Document Status**: PROVISIONAL
**Review Date**: 2026-05-01
**Next Update**: After Phase 2C analyzer renames and algorithm fixes
