# Architecture Map

## Runtime Spine

The core runtime path is:

1. `src/raggov/cli.py`
2. `src/raggov/engine.py`
3. default analyzer suite in `DiagnosisEngine._default_analyzers()`
4. result attachment into `run.metadata`
5. primary failure selection in `src/raggov/decision_policy.py`
6. diagnosis assembly into `src/raggov/models/diagnosis.py`

## Default Analyzer Order

From `DiagnosisEngine._default_analyzers()` (5-layer execution order):

### LAYER 1 — INTAKE GATE
1. `ParserValidationAnalyzer`
2. `SufficiencyAnalyzer`

### LAYER 2 — CLAIM / EVIDENCE EXTRACTION
3. `ClaimGroundingAnalyzer`
4. `ClaimAwareSufficiencyAnalyzer`

### LAYER 3 — RETRIEVAL HEALTH AGGREGATION
5. `ScopeViolationAnalyzer`
6. `StaleRetrievalAnalyzer`
7. `CitationFaithfulnessAnalyzerV0`
8. `CitationMismatchAnalyzer`
9. `InconsistentChunksAnalyzer`
10. `TemporalSourceValidityAnalyzerV1`
11. `RetrievalDiagnosisAnalyzerV0`
12. `CitationFaithfulnessProbe`
13. `RetrievalAnomalyAnalyzer`

### LAYER 4 — SECURITY
14. `PromptInjectionAnalyzer`
15. `PoisoningHeuristicAnalyzer`
16. `PrivacyAnalyzer`

### LAYER 5 — ATTRIBUTION + CONFIDENCE
17. `Layer6TaxonomyClassifier`
18. `SemanticEntropyAnalyzer`

### Optional Analyzers (enabled via config)
- `NCVPipelineVerifier` — inserted after Layer 3 when enabled
- `A2PAttributionAnalyzer` — appended when enabled

## Core Layers

### Evidence-Producing Substrates

- `src/raggov/analyzers/grounding/`
- `src/raggov/analyzers/sufficiency/`
- `src/raggov/analyzers/retrieval/`
- `src/raggov/analyzers/citation_faithfulness/analyzer.py`
- `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
- `src/raggov/analyzers/version_validity/analyzer.py`
- `src/raggov/analyzers/security/`
- `src/raggov/analyzers/parsing/parser_validation.py`

### Shared Evidence Contracts

- `src/raggov/models/grounding.py`
- `src/raggov/models/retrieval_evidence.py`
- `src/raggov/models/citation_faithfulness.py`
- `src/raggov/models/retrieval_diagnosis.py`
- `src/raggov/models/version_validity.py`
- `src/raggov/models/ncv.py`
- `src/raggov/models/pinpoint.py`
- `src/raggov/models/diagnosis.py`
- `src/raggov/models/run.py`

### Meta Interpretation Layers

- `src/raggov/analyzers/verification/ncv.py`
- `src/raggov/analyzers/taxonomy_classifier/layer6.py`
- `src/raggov/analyzers/attribution/a2p.py`
- `src/raggov/analyzers/confidence/semantic_entropy.py`
- `src/raggov/decision_policy.py`

### Optional External Signal Layer

- `src/raggov/evaluators/registry.py`
- `src/raggov/evaluators/retrieval/`
- `src/raggov/evaluators/claim/`
- `src/raggov/evaluators/citation/`
- `src/raggov/external_signal_bridge.py`

External evaluators are advisory.
GovRAG owns diagnosis and final policy.

## Actual Dependency Direction

The strongest dependency chain in the codebase is:

1. claim extraction
2. evidence candidate selection
3. claim grounding
4. citation faithfulness and citation probe
5. sufficiency and claim-aware sufficiency
6. retrieval diagnosis and version validity
7. NCV / Layer6 / A2P / semantic entropy proxy
8. final decision policy

This is why the architecture should be described as diagnosis-first and substrate-first.
