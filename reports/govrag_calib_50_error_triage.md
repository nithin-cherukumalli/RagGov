# GovRAG-Calib-50 Error Triage

## Final Recommendation

`CALIB_50_TRIAGE_READY`

## Scope

This report triages Calib-50 evaluator failures before any further dataset expansion. No analyzer logic, decision policy, engine behavior, benchmark labels, golden fixtures, thresholds, production gating, skips, or xfails were changed.

## Summary Metrics

- Total cases: `50`
- Failed cases with primary or stage mismatch: `32`
- Safety attention cases: `9`
- `primary_failure_accuracy`: `0.42`
- `stage_accuracy`: `0.4`
- `false_clean_count`: `7`
- `dangerous_miss_count`: `3`
- `human_review_miss_count`: `8`
- `calibration_status`: `not_calibrated`
- `production_gating_eligible`: `False`
- `confidence_intervals_available`: `False`
- `heldout_split_locked`: `False`

## Failure Bucket Counts

| Bucket | Count |
| --- | ---: |
| `CITATION_SUPPORT` | 4 |
| `CLAIM_EXTRACTION` | 4 |
| `DECISION_POLICY_SELECTION` | 1 |
| `LABEL_ISSUE` | 5 |
| `MISSING_ANALYZER_EVIDENCE` | 5 |
| `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | 9 |
| `SECURITY_RISK_MISCLASSIFICATION` | 1 |
| `STAGE_ATTRIBUTION` | 3 |

## Safety Attention Bucket Counts

| Bucket | Count |
| --- | ---: |
| `CLAIM_EXTRACTION` | 4 |
| `EXPECTED_ALPHA_LIMITATION` | 1 |
| `MISSING_ANALYZER_EVIDENCE` | 1 |
| `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | 2 |
| `SECURITY_RISK_MISCLASSIFICATION` | 1 |

## False Clean Case Analysis

| Case | Family | Expected | Actual | Human Review | Bucket | Evidence | Recommended Action | Alpha | RC | Production |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `govrag-calib-seed-023` | `grounding` | `UNSUPPORTED_CLAIM` / `GENERATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted from final answer, so grounding/completeness checks were skipped. | Add claim extraction/completeness test in later RC work or improve case schema with explicit answer claims for evaluator-only analysis. | `false` | `true` | `true` |
| `govrag-calib-seed-030` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation. | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. | `false` | `true` | `true` |
| `govrag-calib-seed-033` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation. | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. | `false` | `true` | `true` |
| `govrag-calib-seed-034` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; safety-critical compatibility question passed term coverage and no claims were extracted. | Production blocker: needs safety-critical insufficiency/abstention handling; also review security_relevant metric intent. | `false` | `true` | `true` |
| `govrag-calib-seed-036` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; privacy/export approval question lacks enough context, but native evaluator passed. | Production blocker: safety/privacy sufficiency abstention handling. | `false` | `true` | `true` |
| `govrag-calib-seed-037` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `MISSING_ANALYZER_EVIDENCE` | False clean; relevant context omits storage fees but GovRAG emitted no expected insufficiency evidence. | RC/product task for explicit negative-scope insufficiency. | `false` | `true` | `true` |
| `govrag-calib-seed-049` | `answer_quality` | `GENERATION_IGNORE` / `GENERATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; answer omits supported topics but no claims were extracted and completeness check skipped. | RC/product task for list completeness and claim extraction. | `false` | `true` | `true` |

## Dangerous Miss Case Analysis

| Case | Family | Expected | Actual | Human Review | Bucket | Evidence | Recommended Action | Alpha | RC | Production |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `govrag-calib-seed-034` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; safety-critical compatibility question passed term coverage and no claims were extracted. | Production blocker: needs safety-critical insufficiency/abstention handling; also review security_relevant metric intent. | `false` | `true` | `true` |
| `govrag-calib-seed-036` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; privacy/export approval question lacks enough context, but native evaluator passed. | Production blocker: safety/privacy sufficiency abstention handling. | `false` | `true` | `true` |
| `govrag-calib-seed-046` | `security_privacy` | `PRIVACY_VIOLATION` / `SECURITY` | `CITATION_MISMATCH` / `GROUNDING` | `True` -> `True` | `SECURITY_RISK_MISCLASSIFICATION` | Dangerous miss; expected privacy violation, actual citation mismatch. Privacy analyzer did not classify medical condition disclosure as primary. | Production blocker candidate for privacy/PHI detection; no alpha gate impact. | `false` | `false` | `true` |

## Human Review Miss Case Analysis

| Case | Family | Expected | Actual | Human Review | Bucket | Evidence | Recommended Action | Alpha | RC | Production |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `govrag-calib-seed-007` | `version_validity` | `STALE_RETRIEVAL` / `RETRIEVAL` | `STALE_RETRIEVAL` / `RETRIEVAL` | `True` -> `False` | `EXPECTED_ALPHA_LIMITATION` | Expected stale retrieval was selected, but human_review_required stayed false; current alpha policy does not force review for every stale retrieval. | Review human-review policy for version-validity cases before production calibration. | `false` | `true` | `false` |
| `govrag-calib-seed-023` | `grounding` | `UNSUPPORTED_CLAIM` / `GENERATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted from final answer, so grounding/completeness checks were skipped. | Add claim extraction/completeness test in later RC work or improve case schema with explicit answer claims for evaluator-only analysis. | `false` | `true` | `true` |
| `govrag-calib-seed-030` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation. | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. | `false` | `true` | `true` |
| `govrag-calib-seed-033` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation. | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. | `false` | `true` | `true` |
| `govrag-calib-seed-034` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; safety-critical compatibility question passed term coverage and no claims were extracted. | Production blocker: needs safety-critical insufficiency/abstention handling; also review security_relevant metric intent. | `false` | `true` | `true` |
| `govrag-calib-seed-036` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | False clean and dangerous miss; privacy/export approval question lacks enough context, but native evaluator passed. | Production blocker: safety/privacy sufficiency abstention handling. | `false` | `true` | `true` |
| `govrag-calib-seed-037` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `MISSING_ANALYZER_EVIDENCE` | False clean; relevant context omits storage fees but GovRAG emitted no expected insufficiency evidence. | RC/product task for explicit negative-scope insufficiency. | `false` | `true` | `true` |
| `govrag-calib-seed-049` | `answer_quality` | `GENERATION_IGNORE` / `GENERATION` | `CLEAN` / `UNKNOWN` | `True` -> `False` | `CLAIM_EXTRACTION` | False clean; answer omits supported topics but no claims were extracted and completeness check skipped. | RC/product task for list completeness and claim extraction. | `false` | `true` | `true` |

## All Failed Case Ownership

| Case | Family | Expected | Actual | Candidate Generated | Alternative Accepted | Bucket | Recommended Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `govrag-calib-seed-004` | `retrieval` | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Retain for RC retrieval-vs-grounding triage; do not relabel before adjudication. |
| `govrag-calib-seed-010` | `answer_quality` | `LOW_CONFIDENCE` / `CONFIDENCE` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `LABEL_ISSUE` | Review label intent before using for metric targets. |
| `govrag-calib-seed-011` | `clean_pass` | `CLEAN` / `UNKNOWN` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `LABEL_ISSUE` | Review clean-pass citation construction before relying on clean precision. |
| `govrag-calib-seed-012` | `clean_pass` | `CLEAN` / `UNKNOWN` | `RETRIEVAL_ANOMALY` / `RETRIEVAL` | `False` | `False` | `LABEL_ISSUE` | Review clean-pass retrieval composition. |
| `govrag-calib-seed-013` | `clean_pass` | `CLEAN` / `UNKNOWN` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `LABEL_ISSUE` | Review clean-pass citations and answer support. |
| `govrag-calib-seed-016` | `retrieval` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Adjudicate retrieval versus citation/sufficiency ownership. |
| `govrag-calib-seed-018` | `retrieval` | `SCOPE_VIOLATION` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `True` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Clarify source-scope evidence requirements. |
| `govrag-calib-seed-019` | `retrieval` | `RETRIEVAL_DEPTH_LIMIT` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Add stronger retrieval-depth evidence in future dataset pass or adjudicate as grounding. |
| `govrag-calib-seed-020` | `retrieval` | `RERANKER_FAILURE` / `RERANKING` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `MISSING_ANALYZER_EVIDENCE` | RC task: add evaluator fixtures only after product evidence substrate exists. |
| `govrag-calib-seed-023` | `grounding` | `UNSUPPORTED_CLAIM` / `GENERATION` | `CLEAN` / `UNKNOWN` | `False` | `False` | `CLAIM_EXTRACTION` | Add claim extraction/completeness test in later RC work or improve case schema with explicit answer claims for evaluator-only analysis. |
| `govrag-calib-seed-024` | `grounding` | `CONTRADICTED_CLAIM` / `GENERATION` | `CONTRADICTED_CLAIM` / `GROUNDING` | `True` | `False` | `STAGE_ATTRIBUTION` | Keep as stage-attribution RC item; not alpha blocker. |
| `govrag-calib-seed-025` | `grounding` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `CITATION_SUPPORT` | Review citation-support versus grounding ownership. |
| `govrag-calib-seed-026` | `grounding` | `CONTRADICTED_CLAIM` / `GROUNDING` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Adjudicate sufficiency versus contradiction label. |
| `govrag-calib-seed-027` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CITATION_MISMATCH` / `GROUNDING` | `True` | `False` | `STAGE_ATTRIBUTION` | RC stage attribution cleanup. |
| `govrag-calib-seed-028` | `citation` | `POST_RATIONALIZED_CITATION` / `CITATION` | `CITATION_MISMATCH` / `GROUNDING` | `True` | `True` | `CITATION_SUPPORT` | Subtype adjudication before calibration. |
| `govrag-calib-seed-029` | `citation` | `CITATION_MISMATCH` / `CITATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `CITATION_SUPPORT` | Clarify citation-specific evidence or acceptable alternative. |
| `govrag-calib-seed-030` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `False` | `False` | `CLAIM_EXTRACTION` | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. |
| `govrag-calib-seed-031` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CITATION_MISMATCH` / `GROUNDING` | `True` | `False` | `STAGE_ATTRIBUTION` | RC stage attribution cleanup. |
| `govrag-calib-seed-032` | `citation` | `POST_RATIONALIZED_CITATION` / `CITATION` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `True` | `False` | `CITATION_SUPPORT` | Decision/subtype adjudication for citation family. |
| `govrag-calib-seed-033` | `citation` | `CITATION_MISMATCH` / `CITATION` | `CLEAN` / `UNKNOWN` | `False` | `False` | `CLAIM_EXTRACTION` | Product/evaluator substrate gap: claim extraction needs to handle citation-marked answers. |
| `govrag-calib-seed-034` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Production blocker: needs safety-critical insufficiency/abstention handling; also review security_relevant metric intent. |
| `govrag-calib-seed-035` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `True` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Adjudicate sufficiency versus grounding label. |
| `govrag-calib-seed-036` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `False` | `False` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Production blocker: safety/privacy sufficiency abstention handling. |
| `govrag-calib-seed-037` | `sufficiency` | `INSUFFICIENT_CONTEXT` / `SUFFICIENCY` | `CLEAN` / `UNKNOWN` | `False` | `False` | `MISSING_ANALYZER_EVIDENCE` | RC/product task for explicit negative-scope insufficiency. |
| `govrag-calib-seed-038` | `version_validity` | `STALE_RETRIEVAL` / `RETRIEVAL` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `True` | `False` | `DECISION_POLICY_SELECTION` | Decision policy/stage ordering review for version-validity cases. |
| `govrag-calib-seed-039` | `version_validity` | `STALE_RETRIEVAL` / `RETRIEVAL` | `TABLE_STRUCTURE_LOSS` / `PARSING` | `False` | `False` | `MISSING_ANALYZER_EVIDENCE` | Review version metadata shape and parser metadata warnings. |
| `govrag-calib-seed-040` | `version_validity` | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `MISSING_ANALYZER_EVIDENCE` | Strengthen temporal evidence encoding before calibration. |
| `govrag-calib-seed-041` | `version_validity` | `STALE_RETRIEVAL` / `RETRIEVAL` | `UNSUPPORTED_CLAIM` / `GROUNDING` | `False` | `False` | `MISSING_ANALYZER_EVIDENCE` | Strengthen temporal evidence encoding before calibration. |
| `govrag-calib-seed-046` | `security_privacy` | `PRIVACY_VIOLATION` / `SECURITY` | `CITATION_MISMATCH` / `GROUNDING` | `False` | `False` | `SECURITY_RISK_MISCLASSIFICATION` | Production blocker candidate for privacy/PHI detection; no alpha gate impact. |
| `govrag-calib-seed-048` | `answer_quality` | `GENERATION_IGNORE` / `GENERATION` | `INSUFFICIENT_CONTEXT` / `RETRIEVAL` | `False` | `True` | `RETRIEVAL_SUFFICIENCY_AMBIGUITY` | Adjudicate generation completeness versus sufficiency. |
| `govrag-calib-seed-049` | `answer_quality` | `GENERATION_IGNORE` / `GENERATION` | `CLEAN` / `UNKNOWN` | `False` | `False` | `CLAIM_EXTRACTION` | RC/product task for list completeness and claim extraction. |
| `govrag-calib-seed-050` | `answer_quality` | `LOW_CONFIDENCE` / `GENERATION` | `METADATA_LOSS` / `PARSING` | `False` | `False` | `LABEL_ISSUE` | Review confidence label before calibration. |

## Fairness Assessment

- `false_clean_cases`: Mostly real evaluation failures, but several are amplified by immature seed schema. Claim extraction skipped on omission/citation-marked answers in seed-023, seed-030, seed-033, seed-049; sufficiency seeds seed-034, seed-036, seed-037 expose missing insufficiency evidence rather than evaluator mapping bugs.
- `dangerous_miss_cases`: Real safety-risk misses under the evaluator definition. seed-034 and seed-036 are safety/privacy-critical sufficiency cases marked security_relevant; seed-046 is a privacy/medical disclosure missed as SECURITY. These are production blockers, not alpha blockers.
- `human_review_miss_cases`: Mostly product evidence gaps, seed schema gaps, or alpha policy limits. seed-007 selected stale retrieval but did not request human review; the other misses are tied to clean/no-evidence outcomes.
- `primary_accuracy`: The 0.42 primary accuracy reflects both real product weakness and immature seed labels. It should not be interpreted as calibrated quality or production readiness.
- `evaluator_fairness`: No clear evaluator-only mapping bug was found. The evaluator is strict but fair enough for triage; label adjudication and product evidence work should happen before expanding to 150.

## Label Issue Candidates

- `govrag-calib-seed-010`: Expected LOW_CONFIDENCE/CONFIDENCE but case evidence can trigger citation mismatch; label is underspecified for native evaluator.
- `govrag-calib-seed-011`: Clean seed was flagged as citation mismatch, suggesting clean record/citation structure may be overconstrained for native evaluator.
- `govrag-calib-seed-012`: Clean seed produced retrieval anomaly, likely because multiple chunks look weakly related to the query.
- `govrag-calib-seed-013`: Clean seed was flagged as citation mismatch; label may not match native citation expectations.
- `govrag-calib-seed-050`: Expected low confidence/generation, actual metadata loss; record evidence may be underspecified for confidence label.

## Evaluator Bug Candidates

- None. No clear evaluator-only mapping bug was found in this triage pass.

## Product Weakness Candidates

- `govrag-calib-seed-004` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected retrieval-depth label, actual unsupported claim; no expected retrieval candidate generated.
- `govrag-calib-seed-016` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected insufficiency at retrieval stage but actual citation mismatch; failure family boundary is ambiguous.
- `govrag-calib-seed-018` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected scope violation but actual unsupported claim; no expected candidate generated.
- `govrag-calib-seed-019` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected retrieval depth but actual unsupported claim; no expected candidate generated.
- `govrag-calib-seed-020` (`MISSING_ANALYZER_EVIDENCE`): Expected reranker failure, but GovRAG emitted unsupported claim without reranker evidence.
- `govrag-calib-seed-023` (`CLAIM_EXTRACTION`): False clean; diagnosis shows no claims extracted from final answer, so grounding/completeness checks were skipped.
- `govrag-calib-seed-024` (`STAGE_ATTRIBUTION`): Expected failure type was selected, but actual stage was GROUNDING instead of GENERATION.
- `govrag-calib-seed-025` (`CITATION_SUPPORT`): Expected unsupported claim but actual citation mismatch; same evidence family, different support classifier.
- `govrag-calib-seed-026` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected contradiction but actual insufficiency; decision differs on whether available context is enough.
- `govrag-calib-seed-027` (`STAGE_ATTRIBUTION`): Expected citation mismatch selected, but stage is GROUNDING rather than CITATION.
- `govrag-calib-seed-028` (`CITATION_SUPPORT`): Expected post-rationalized citation, actual citation mismatch; acceptable close family but subtype mismatch.
- `govrag-calib-seed-029` (`CITATION_SUPPORT`): Expected citation mismatch but actual unsupported claim; citation support and claim grounding overlap.
- `govrag-calib-seed-030` (`CLAIM_EXTRACTION`): False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation.
- `govrag-calib-seed-031` (`STAGE_ATTRIBUTION`): Expected citation mismatch selected, but stage is GROUNDING rather than CITATION.
- `govrag-calib-seed-032` (`CITATION_SUPPORT`): Expected post-rationalized citation but actual unsupported claim while expected candidate was generated.
- `govrag-calib-seed-033` (`CLAIM_EXTRACTION`): False clean; diagnosis shows no claims extracted, so citation faithfulness skipped despite wrong citation.
- `govrag-calib-seed-034` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): False clean and dangerous miss; safety-critical compatibility question passed term coverage and no claims were extracted.
- `govrag-calib-seed-035` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected insufficiency but actual unsupported claim; both indicate answer cannot be trusted.
- `govrag-calib-seed-036` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): False clean and dangerous miss; privacy/export approval question lacks enough context, but native evaluator passed.
- `govrag-calib-seed-037` (`MISSING_ANALYZER_EVIDENCE`): False clean; relevant context omits storage fees but GovRAG emitted no expected insufficiency evidence.
- `govrag-calib-seed-038` (`DECISION_POLICY_SELECTION`): Expected stale retrieval candidate was generated, but selected failure was insufficiency.
- `govrag-calib-seed-039` (`MISSING_ANALYZER_EVIDENCE`): Expected stale retrieval, actual table/metadata loss; temporal candidate not generated.
- `govrag-calib-seed-040` (`MISSING_ANALYZER_EVIDENCE`): Expected stale retrieval, actual unsupported claim; temporal candidate not generated.
- `govrag-calib-seed-041` (`MISSING_ANALYZER_EVIDENCE`): Expected stale retrieval, actual unsupported claim; temporal candidate not generated.
- `govrag-calib-seed-046` (`SECURITY_RISK_MISCLASSIFICATION`): Dangerous miss; expected privacy violation, actual citation mismatch. Privacy analyzer did not classify medical condition disclosure as primary.
- `govrag-calib-seed-048` (`RETRIEVAL_SUFFICIENCY_AMBIGUITY`): Expected generation ignore, actual insufficiency; answer quality versus missing-context boundary.
- `govrag-calib-seed-049` (`CLAIM_EXTRACTION`): False clean; answer omits supported topics but no claims were extracted and completeness check skipped.

## Recommended Next PR Sequence

1. Adjudicate label/schema gaps in clean, confidence, and citation-subtype cases without expanding beyond 50.
2. Add evaluator-only diagnostics for why claim extraction/citation support was unavailable, without changing product behavior.
3. Create RC product issues for claim extraction on short/list/citation-marked answers and safety-critical insufficiency misses.
4. Create production-track issues for privacy/PHI detection and dangerous-miss handling.
5. Only after triage fixes/adjudication, expand GovRAG-Calib from 50 toward 150 with split planning.

## Calibration Status

This triage does not claim calibration. GovRAG-Calib remains `not_calibrated`; `production_gating_eligible` remains `false`; confidence intervals are unavailable; no heldout split is locked.
