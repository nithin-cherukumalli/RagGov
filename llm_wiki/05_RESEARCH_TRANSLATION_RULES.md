# Research Translation Rules

## Hard Rule

Research-inspired is not the same as research-faithful.

Any mention of a paper, framework, or branded method must be tagged as one of:

- faithful implementation
- partial approximation
- conceptual inspiration only
- not implemented yet

## Current Honest Readings

- `SemanticEntropyAnalyzer`
  status: partial approximation
  reason: deterministic path is a claim-label entropy proxy; LLM sampling path is closer but not the default universal path

- `Layer6TaxonomyClassifier`
  status: conceptual inspiration only or partial approximation depending on call path
  reason: rule-based taxonomy mapping, not a trained classifier

- `A2PAttributionAnalyzer`
  status: partial approximation
  reason: evidence-backed in some paths, but still includes deterministic and legacy heuristic fallback modes

- `CitationFaithfulnessAnalyzerV0`
  status: partial approximation
  reason: uses grounding and retrieval evidence, explicitly not a research-faithful RefChecker or RAGChecker implementation

- `RetrievalDiagnosisAnalyzerV0`
  status: conceptual inspiration only
  reason: heuristic rollup over native evidence, explicitly not RAGChecker, RAGAS, DeepEval, RefChecker, Layer6, or A2P

- `TemporalSourceValidityAnalyzerV1`
  status: partial approximation
  reason: explicitly not a research-faithful VersionRAG implementation

## Forbidden Behavior

- Do not use paper names as credibility decoration.
- Do not write “based on paper X” without stating the implementation gap.
- Do not call a heuristic threshold “calibrated” unless labeled data and method support that claim.
