# Manual Inspection Report Template

**Purpose**: Human spot-check of automated diagnosis results before trusting metrics.

**Validation Run**: `[DATE]` - `baseline_validation_v1.json`

---

## Instructions

1. For each case listed below, review the diagnosis output against the expected failure
2. Mark your judgment: AGREE | DISAGREE | UNCERTAIN
3. Add notes explaining your reasoning, especially for edge cases
4. Flag any patterns suggesting systematic analyzer issues

---

## Cases for Manual Inspection

### Case: `[case_id]`

**Run Fixture**: `[path/to/fixture.json]`

**Query**: `[question text]`

**Retrieved Chunks** (first 3):
```
[chunk_id]: [first 100 chars of text]...
[chunk_id]: [first 100 chars of text]...
[chunk_id]: [first 100 chars of text]...
```

**Final Answer** (first 200 chars):
```
[answer text]...
```

**Expected Diagnosis**:
- Primary Failure: `[FAILURE_TYPE]`
- Root Cause Stage: `[STAGE]`
- Should Have Answered: `[true|false]`

**Observed Diagnosis**:
- Primary Failure: `[FAILURE_TYPE]`
- Root Cause Stage: `[STAGE]`
- Should Have Answered: `[true|false]`
- Confidence: `[0.00-1.00]`

**Evidence Snippets**:
```
[first 3 evidence strings from diagnosis]
```

**Human Judgment**: [ ] AGREE  [ ] DISAGREE  [ ] UNCERTAIN

**Notes**:
```
[Your reasoning here]
```

**Pattern Flags** (check all that apply):
- [ ] Analyzer overly sensitive (too many false positives)
- [ ] Analyzer insufficiently sensitive (missed failure)
- [ ] Stage attribution questionable
- [ ] Confidence score does not match evidence
- [ ] Evidence strings unhelpful or misleading
- [ ] Recommended fix not actionable

---

### Case: `[case_id]`

[Repeat template for each case requiring inspection]

---

## Summary

**Total Cases Inspected**: `[N]`

**Agreement Distribution**:
- AGREE: `[N]` cases
- DISAGREE: `[N]` cases
- UNCERTAIN: `[N]` cases

**Human Agreement Rate**: `[N/total]` %

**Recurring Patterns Identified**:

1. **Pattern**: [description]
   - **Affected Analyzer**: [name]
   - **Frequency**: [N cases]
   - **Recommended Action**: [what to fix]

2. **Pattern**: [description]
   - **Affected Analyzer**: [name]
   - **Frequency**: [N cases]
   - **Recommended Action**: [what to fix]

**Analyzers Flagged for Review**:

| Analyzer | Issue | Cases | Action |
|----------|-------|-------|--------|
| [name] | [issue description] | [N] | [recommended fix] |
| [name] | [issue description] | [N] | [recommended fix] |

**Notes on Sample Size Limitations**:
```
[Acknowledge that with only ~10 cases, patterns may not be statistically significant]
[Note which failure types are under-represented in the current fixture set]
[Identify which analyzers cannot be meaningfully evaluated due to lack of relevant test cases]
```

---

## Recommendations for Phase 1B

Based on manual inspection, the following analyzers should be:

**Calibration-Ready** (sufficient agreement, enough test cases):
- [analyzer_name]: [brief justification]

**Provisional** (some agreement, but small sample or edge case sensitivity):
- [analyzer_name]: [brief justification]

**Not Calibration-Ready** (low agreement, systemic issues, or no test coverage):
- [analyzer_name]: [brief justification]

---

## Next Steps

1. [ ] Address critical analyzer issues identified above
2. [ ] Create fixtures for under-represented failure types
3. [ ] Re-run baseline validation after fixes
4. [ ] Proceed to Phase 1B calibration readiness assessment

---

**Inspector**: `[Your Name]`
**Date**: `[YYYY-MM-DD]`
**Validation Confidence**: [ ] HIGH [ ] MEDIUM [ ] LOW
