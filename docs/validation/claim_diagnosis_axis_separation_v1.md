# Claim-Diagnosis Evaluation Semantics Fix: Axis Separation v1

**Date**: 2026-04-28
**Status**: Completed
**Evaluation Status**: `diagnostic_gold_v0_small_unvalidated`

## Executive Summary

Successfully refactored the claim-diagnosis evaluation harness to separate conflated diagnostic concepts into independent evaluation axes. This fix corrects the evaluation contract without changing analyzer implementations, making the benchmark more truthful about actual system capabilities and failures.

## Problem Statement

### Semantic Conflation (Before)

The original evaluation harness conflated multiple independent diagnostic concerns into a single "claim label" field:

1. **Textual claim support** (does the evidence text support the claim?)
2. **Citation validity** (are citations correct?)
3. **Freshness validity** (is the source stale?)

This conflation caused:
- **False negatives**: Cases marked as failed when only citation or freshness was problematic, despite correct claim grounding
- **Ambiguous failures**: Unable to distinguish between different failure modes
- **Misleading metrics**: Single accuracy score masked separate capabilities

### Example Conflations

**Case 1: `stale_source_case`**
- **Before**: Expected claim_label = "unsupported" (conflating textual support with source freshness)
- **Issue**: If retrieved stale text textually supports the claim, should claim_label be "entailed" or "unsupported"?
- **After**: Separate `expected_claim_label` (textual support) from `expected_freshness_validity` (temporal quality)

**Case 2: `citation_mismatch_case`**
- **Before**: Expected claim_label = "unsupported" (conflating textual support with citation correctness)
- **Issue**: If retrieved text supports claim but cited doc_id is phantom, should claim_label be "entailed" or "unsupported"?
- **After**: Separate `expected_claim_label` from `expected_citation_validity`

## Solution: Five Independent Evaluation Axes

### Axis A: Claim Support
**Question**: Does retrieved evidence text support, contradict, or fail to support the claim?

**Labels**: `entailed` | `unsupported` | `contradicted`

**Evaluated using**: ClaimGroundingAnalyzer.claim_results

**Gold field**: `expected_claim_label` (with backward compat to `expected_label`)

### Axis B: Citation Validity
**Question**: Are cited document IDs present in retrieved context, and do citations appear faithful?

**Labels**: `valid` | `invalid` | `not_applicable`

**Evaluated using**:
- CitationMismatchAnalyzer (phantom citations)
- CitationFaithfulnessProbe (post-rationalized citations)

**Gold field**: `expected_citation_validity`

**Default**: Cases without explicit expected_citation_validity pass if observed is `valid` or `not_applicable`

### Axis C: Freshness Validity
**Question**: Is the evidence source current enough / not stale?

**Labels**: `fresh` | `stale` | `unknown`

**Evaluated using**:
- StaleRetrievalAnalyzer
- corpus_entries timestamps

**Gold field**: `expected_freshness_validity`

**Default**: Cases without explicit expected_freshness_validity pass if observed is `fresh` or `unknown`

### Axis D: Context Sufficiency
**Question**: Was there enough evidence to answer or verify claims safely?

**Labels**: `sufficient` (true) | `insufficient` (false)

**Evaluated using**:
- ClaimAwareSufficiencyAnalyzer.sufficiency_result (preferred)
- Fallback to SufficiencyAnalyzer

**Gold field**: `expected_sufficiency` (new) or `expected_sufficient` (backward compat)

### Axis E: A2P Root Cause
**Question**: What is the likely root cause of failed or risky output?

**Labels**: `insufficient_context_or_retrieval_miss` | `weak_or_ambiguous_evidence` | `generation_contradicted_retrieved_evidence` | `stale_source_usage` | `citation_mismatch` | `verification_uncertainty` | `none`

**Evaluated using**: A2PAttributionAnalyzer.claim_attributions

**Gold field**: `expected_a2p_primary_cause`

## Schema Changes

### ClaimExpectation Model
```python
class ClaimExpectation(BaseModel):
    claim_text: str

    # Axis A: Claim Support
    expected_claim_label: str | None = None
    expected_label: str | None = None  # Backward compatibility

    # Axis B: Citation Validity
    expected_citation_validity: str | None = None

    # Axis C: Freshness Validity
    expected_freshness_validity: str | None = None

    # Axis E: A2P Root Cause
    expected_a2p_primary_cause: str | None = None
```

### ClaimDiagnosisGoldCase Model
```python
class ClaimDiagnosisGoldCase(BaseModel):
    # ... existing fields ...

    # Axis D: Context Sufficiency
    expected_sufficient: bool | None = None  # Backward compat
    expected_sufficiency: bool | None = None  # New
```

### ClaimDiagnosisCaseResult Dataclass
```python
@dataclass(frozen=True)
class ClaimDiagnosisCaseResult:
    case_id: str
    claim_label_pass: bool              # Axis A
    citation_validity_pass: bool         # Axis B
    freshness_validity_pass: bool        # Axis C
    sufficiency_pass: bool               # Axis D
    a2p_primary_cause_pass: bool         # Axis E
    # ... stage/fix fields ...
```

### ClaimDiagnosisHarnessResult
```python
@dataclass(frozen=True)
class ClaimDiagnosisHarnessResult:
    claim_label_accuracy: float          # Axis A
    citation_validity_accuracy: float    # Axis B
    freshness_validity_accuracy: float   # Axis C
    sufficiency_accuracy: float          # Axis D
    a2p_primary_cause_accuracy: float    # Axis E
    # ... stage/fix accuracies ...
```

## Gold Data Updates

Updated all 10 cases in `claim_diagnosis_gold_v0.json`:

**General updates**:
- Added `expected_claim_label` to all cases (duplicating `expected_label` for now)
- Maintained backward compatibility by keeping `expected_label`

**Semantic fixes**:

1. **stale_source_case**:
   - `expected_claim_label`: "unsupported" (12 months ≠ 36 months textually)
   - `expected_freshness_validity`: "stale" (source from 2020-01-01)
   - `expected_a2p_primary_cause`: "stale_source_usage"

2. **citation_mismatch_case**:
   - `expected_claim_label`: "unsupported" ($100 ≠ $500 textually)
   - `expected_citation_validity`: "invalid" (phantom doc-phantom-1)
   - `expected_a2p_primary_cause`: "citation_mismatch"

## Evaluator Changes

### Extraction Functions
Added helper functions to extract observed axis values:

```python
def _observed_citation_validity(diagnosis: Diagnosis) -> str:
    """Returns: 'invalid' | 'valid' | 'not_applicable'"""

def _observed_freshness_validity(diagnosis: Diagnosis) -> str:
    """Returns: 'stale' | 'fresh' | 'unknown'"""
```

### Evaluation Logic
Updated `_evaluate_example` to:
1. Check `expected_claim_label` with fallback to `expected_label`
2. Evaluate citation validity separately
3. Evaluate freshness validity separately
4. Use `expected_sufficiency` with fallback to `expected_sufficient`
5. Default pass for not_applicable cases

### Report Updates

**Markdown report** now includes:
```
Aggregate Metrics (by axis):
  Axis A - Claim Support:       0.80
  Axis B - Citation Validity:   1.00
  Axis C - Freshness Validity:  1.00
  Axis D - Context Sufficiency: 0.70
  Axis E - A2P Root Cause:      0.70

Mismatches by Axis:

Axis A - Claim Support (2 mismatches):
  - stale_source_case: ...
  - citation_mismatch_case: ...

Axis D - Context Sufficiency (3 mismatches):
  - unsupported_missing_1: ...
```

**JSON report** now includes per-case:
```json
{
  "citation_validity_pass": true,
  "freshness_validity_pass": true,
  ...
}
```

## Baseline Results

### Before vs After Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Claim Support (A) | 0.80 | 8/10 cases pass |
| Citation Validity (B) | 1.00 | All cases pass (none explicitly test citations) |
| Freshness Validity (C) | 1.00 | All cases pass (only 1 tests staleness) |
| Context Sufficiency (D) | 0.70 | 7/10 cases pass |
| A2P Root Cause (E) | 0.70 | 7/10 cases pass |
| Primary Stage | 0.90 | 9/10 cases pass |
| Fix Category (partial) | 0.90 | 9/10 cases pass |

### Remaining Mismatches (3 cases)

#### 1. unsupported_missing_1
**Type**: Real system bug in sufficiency analyzer

**Expected**:
- Sufficiency: False
- A2P cause: insufficient_context_or_retrieval_miss
- Stage: RETRIEVAL

**Observed**:
- Sufficiency: True (WRONG - c1 doesn't support weekend phone coverage)
- A2P cause: weak_or_ambiguous_evidence
- Stage: GROUNDING

**Root cause**: ClaimAwareSufficiencyAnalyzer incorrectly interprets `supporting_chunk_ids=["c1"]` as sufficient, even though the claim is marked "unsupported". This reveals a contract ambiguity in `ClaimResult.supporting_chunk_ids`:
- Does it mean "chunks that SUPPORT the claim"?
- Or "best candidate/related chunks for evidence"?

**Fix needed**: Clarify supporting_chunk_ids contract or update ClaimAwareSufficiencyAnalyzer logic.

## Value-Aware Semantics Alignment (2026-04-28 Update)

After introducing deterministic value-aware grounding v1, claim-label semantics were aligned as follows:

- `entailed`: retrieved evidence supports claim facts and critical values.
- `unsupported`: evidence does not establish claim and no direct conflicting value is stated.
- `contradicted`: evidence states a conflicting value/date/duration/percent/threshold/GO number for the same attribute.

Axis separation remains intentional:

- A case can be **claim contradicted** and **freshness stale** simultaneously.
- A case can be **claim contradicted** and **citation invalid** simultaneously.

Current v1 A2P remains single-primary-cause. When direct claim contradiction exists, A2P may prioritize generation-level contradiction over stale/citation causes, while stale/citation axes are still evaluated independently.

#### 2. stale_source_case
**Type**: Real system bug in claim grounding analyzer

**Expected**:
- Claim label: "unsupported" (12 months ≠ 36 months)
- Sufficiency: False
- A2P cause: stale_source_usage

**Observed**:
- Claim label: "entailed" (WRONG - numeric anchor mismatch)
- Sufficiency: True (WRONG - follows from grounding error)
- A2P cause: None (no attribution generated)

**Root cause**: ClaimGroundingAnalyzer's structured verifier is not catching the numeric mismatch between "twelve months" and "thirty six months". This is a bug in either:
- Anchor extraction (not finding "twelve"/"thirty six")
- Numeric normalization
- Anchor matching threshold

**Fix needed**: Improve numeric anchor extraction/matching in ClaimGroundingAnalyzer.

#### 3. citation_mismatch_case
**Type**: Real system bug in claim grounding analyzer

**Expected**:
- Claim label: "unsupported" ($100 ≠ $500)
- Sufficiency: False
- A2P cause: citation_mismatch

**Observed**:
- Claim label: "entailed" (WRONG - numeric anchor mismatch)
- Sufficiency: True (WRONG - follows from grounding error)
- A2P cause: None (no attribution generated)

**Root cause**: Same as stale_source_case - ClaimGroundingAnalyzer not catching numeric differences.

**Fix needed**: Same as stale_source_case.

## What This Task Fixed

### Evaluation Contract Issues (Fixed)
1. ✅ Separated claim textual support from citation validity
2. ✅ Separated claim textual support from freshness validity
3. ✅ Added independent axes for all diagnostic concerns
4. ✅ Made evaluation metrics more granular and truthful
5. ✅ Enabled diagnosis of which specific capabilities fail

### Gold/Evaluation Semantic Issues (Fixed)
1. ✅ stale_source_case can now express: claim textually unsupported AND source stale
2. ✅ citation_mismatch_case can now express: claim textually unsupported AND citation invalid
3. ✅ Removed false conflation of different failure types

## What Should Be Fixed Next

### Real System Bugs (Not Fixed - Out of Scope)
1. ❌ ClaimGroundingAnalyzer numeric anchor matching
2. ❌ ClaimAwareSufficiencyAnalyzer supporting_chunk_ids contract ambiguity
3. ❌ A2P attribution not detecting stale_source_usage or citation_mismatch

### Recommended Next Steps

1. **Fix ClaimGroundingAnalyzer numeric matching** (HIGH PRIORITY)
   - Improve anchor extraction for numbers
   - Add fuzzy numeric matching threshold
   - Test cases: stale_source_case, citation_mismatch_case

2. **Clarify supporting_chunk_ids contract** (MEDIUM PRIORITY)
   - Document whether it means "supporting" or "related"
   - Update ClaimAwareSufficiencyAnalyzer accordingly
   - Test case: unsupported_missing_1

3. **Extend A2P attribution coverage** (MEDIUM PRIORITY)
   - Add stale_source_usage detection
   - Add citation_mismatch detection
   - These are currently not being attributed

4. **Add more axis-specific test cases** (LOW PRIORITY)
   - Currently only 1 case tests freshness explicitly
   - Currently only 1 case tests citations explicitly
   - Add cases where claim IS entailed but citation/freshness fails

## Production Validity Status

**Still**: `diagnostic_gold_v0_small_unvalidated`

This task improved evaluation truthfulness but did not:
- Fix the underlying analyzer bugs
- Validate gold labels against production RAG runs
- Expand test coverage beyond 10 cases

## Conclusion

The evaluation harness now correctly separates independent diagnostic axes, revealing that the actual system bugs are in:
1. Numeric anchor matching (ClaimGroundingAnalyzer)
2. Sufficiency contract interpretation (ClaimAwareSufficiencyAnalyzer)

These were previously hidden by semantic conflation in the evaluation contract. The benchmark is now more truthful about what actually works and what doesn't.
