# Analyzer Calibration Pre-Registration

## Target analyzers

- `RetrievalAnomalyAnalyzer`
- `CitationFaithfulnessAnalyzerV0`
- `RetrievalDiagnosisAnalyzerV0`
- `ScopeViolationAnalyzer`

## Predicates to change

- `RetrievalAnomalyAnalyzer`: promote retrieval anomaly evidence to
  `status="fail"` when the evidence contains near-duplicate chunk evidence.
- `CitationFaithfulnessAnalyzerV0`: promote missing-citation claim evidence to
  `status="fail"` only when the run has no citations and the citation absence
  is isolated: supporting chunk evidence exists, one retrieved chunk,
  non-abstention answer, and no retrieval profile noise.
- `RetrievalDiagnosisAnalyzerV0`: suppress retrieval-noise warnings when
  `noisy_chunk_ids` do not satisfy the existing `noise_ratio_warn` and
  `noise_min_chunks` predicate.
- `ScopeViolationAnalyzer`: suppress profile-path scope warnings when relevant
  chunks are present and irrelevant chunks do not outnumber relevant chunks.

## Exact code locations

- `src/raggov/analyzers/security/anomalies.py:48`
- `src/raggov/analyzers/citation_faithfulness/analyzer.py:138`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py:317`
- `src/raggov/analyzers/retrieval/scope.py:94`

## Acceptance criteria

- Protected common baseline: 42/46 with composition unchanged
- Protected false_clean_count = 0, false_security_count = 0, false_incomplete_count = 0
- Calib-50 false_clean_count = 0, dangerous_clean_miss_count = 0, human_review_miss_count = 0
- Calib-50 primary_failure_accuracy >= 0.48 (must not regress)
- Set A (011, 012, 013): primary_failure must become CLEAN
- Set B (11, 26, 39): the target analyzer must reach status="fail" (not "warn"),
  AND primary_failure must remain the same family as before (RETRIEVAL_ANOMALY / CITATION_MISMATCH / CITATION_MISMATCH)
- Heldout heldout_v0_1: primary_failure_accuracy >= 0.60 (must not regress)
- On the 13 non-clean heldout cases: no case's primary_failure value may change

## What I will NOT do

- No golden label changes (stresslab/cases/golden/rag_failures.py)
- No new acceptable-alternative entries in scripts/check_protected_baseline.py
- No engine.py changes (the warn-promotion stays for now)
- No new analyzers
- No new dataset cases
- No threshold changes without per-case rationale in the pre-registration
