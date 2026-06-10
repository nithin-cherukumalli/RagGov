# Retrieval Diagnosis Substrate

Core files:

- `src/raggov/analyzers/retrieval/evidence_profile.py`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
- `src/raggov/models/retrieval_evidence.py`
- `src/raggov/models/retrieval_diagnosis.py`

## Current Honest Description

Current retrieval diagnosis is a heuristic rollup over upstream signals.

It does not recompute retrieval truth from first principles.
It does not equal RAGAS, DeepEval, RAGChecker, or RefChecker.

## Evidence Producers

- `RetrievalEvidenceProfilerV0`
- `ScopeViolationAnalyzer`
- `StaleRetrievalAnalyzer`
- `CitationMismatchAnalyzer`
- `InconsistentChunksAnalyzer`
- `RetrievalDiagnosisAnalyzerV0`
- `TemporalSourceValidityAnalyzerV1`

## Key Rule

External retrieval evaluator outputs are advisory probes, not native diagnoses.

## Main Weakness

The retrieval substrate is still heavily heuristic:

- lexical overlap relevance
- age-threshold freshness
- negation-pair contradiction
- rollup inference from multiple imperfect reports
