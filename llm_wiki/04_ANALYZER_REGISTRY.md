# Analyzer Registry

Classification labels used here:

- `deterministic_check`
- `heuristic_baseline`
- `practical_approximation`
- `statistical_signal`
- `learned_signal`
- `research_faithful`
- `experimental`
- `deprecated`

## Core Analyzer Inventory

### Parsing

- `ParserValidationAnalyzer` in `src/raggov/analyzers/parsing/parser_validation.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: rich profile path plus explicit text-only fallback

### Grounding

- `ClaimGroundingAnalyzer` in `src/raggov/analyzers/grounding/support.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: core substrate, supports heuristic and LLM-backed verifier modes, exposes fallback state

- `CitationFaithfulnessProbe` in `src/raggov/analyzers/grounding/citation_faithfulness.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: probe over claim evidence, not full faithfulness science

### Sufficiency

- `SufficiencyAnalyzer` in `src/raggov/analyzers/sufficiency/sufficiency.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: requirement-aware plus deterministic fallback, explicitly advisory

- `ClaimAwareSufficiencyAnalyzer` in `src/raggov/analyzers/sufficiency/claim_aware.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: depends on prior claim evidence

### Retrieval

- `ScopeViolationAnalyzer` in `src/raggov/analyzers/retrieval/scope.py`
  classification: `heuristic_baseline`
  role: evidence producer

- `StaleRetrievalAnalyzer` in `src/raggov/analyzers/retrieval/stale.py`
  classification: `heuristic_baseline`
  role: evidence producer

- `CitationMismatchAnalyzer` in `src/raggov/analyzers/retrieval/citation.py`
  classification: `heuristic_baseline`
  role: evidence producer
  notes: supporting retrieval provenance signal only, not primary grounding-stage citation authority when `CitationFaithfulnessAnalyzerV0` has structured evidence

- `InconsistentChunksAnalyzer` in `src/raggov/analyzers/retrieval/inconsistency.py`
  classification: `heuristic_baseline`
  role: evidence producer

- `RetrievalEvidenceProfilerV0` in `src/raggov/analyzers/retrieval/evidence_profile.py`
  classification: `heuristic_baseline`
  role: evidence producer
  notes: internal utility analyzer, builds retrieval evidence profile for downstream consumers

- `RetrievalDiagnosisAnalyzerV0` in `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py`
  classification: `heuristic_baseline`
  role: evidence producer and rollup
  notes: aggregates upstream reports rather than recomputing retrieval quality

- `ContradictionAnalyzer` in `src/raggov/analyzers/retrieval/contradiction.py`
  classification: `heuristic_baseline`
  role: evidence producer
  notes: detects contradictory chunk pairs

- `FreshnessAnalyzer` in `src/raggov/analyzers/retrieval/freshness.py`
  classification: `heuristic_baseline`
  role: evidence producer
  notes: evaluates document freshness signals

- `RelevanceAnalyzer` in `src/raggov/analyzers/retrieval/relevance.py`
  classification: `heuristic_baseline`
  role: evidence producer
  notes: evaluates query-chunk relevance

### Citation

- `CitationFaithfulnessAnalyzerV0` in `src/raggov/analyzers/citation_faithfulness/analyzer.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: primary citation authority in native mode; uses existing claim grounding and retrieval evidence only

### Version / Temporal Validity

- `TemporalSourceValidityAnalyzerV1` in `src/raggov/analyzers/version_validity/analyzer.py`
  classification: `practical_approximation`
  role: evidence producer

- `VersionValidityAnalyzerV1` in `src/raggov/analyzers/version_validity/analyzer.py`
  classification: `practical_approximation`
  role: evidence producer
  notes: distinct analyzer sharing implementation file with TemporalSourceValidityAnalyzerV1; both are first-class analyzers

### Security

- `PromptInjectionAnalyzer` in `src/raggov/analyzers/security/injection.py`
  classification: `practical_approximation`
  role: evidence producer

- `PoisoningHeuristicAnalyzer` in `src/raggov/analyzers/security/poisoning.py`
  classification: `heuristic_baseline`
  role: evidence producer

- `PrivacyAnalyzer` in `src/raggov/analyzers/security/privacy.py`
  classification: `heuristic_baseline`
  role: evidence producer

- `RetrievalAnomalyAnalyzer` in `src/raggov/analyzers/security/anomalies.py`
  classification: `heuristic_baseline`
  role: evidence producer

### Meta / Interpretation

- `NCVPipelineVerifier` in `src/raggov/analyzers/verification/ncv.py`
  classification: `practical_approximation`
  role: evidence interpreter
  notes: node-wise pipeline verification with explicit heuristic fallbacks

- `Layer6TaxonomyClassifier` in `src/raggov/analyzers/taxonomy_classifier/layer6.py`
  classification: `practical_approximation`
  role: evidence interpreter
  notes: rule-based taxonomy mapping over prior analyzer outputs

- `A2PAttributionAnalyzer` in `src/raggov/analyzers/attribution/a2p.py`
  classification: `experimental`
  role: evidence interpreter
  notes: counterfactual attribution with claim-level and deterministic fallback modes

- `SemanticEntropyAnalyzer` in `src/raggov/analyzers/confidence/semantic_entropy.py`
  classification: `statistical_signal`
  role: evidence interpreter
  notes: LLM sampling path is closer to research-faithful; deterministic path is only a proxy

## External Evaluator / Adapter Inventory

- `RAGASAdapter`
- `DeepEvalAdapter`
- `RAGCheckerSignalProvider`
- `CrossEncoderRetrievalRelevanceProvider`
- `RefCheckerClaimAdapter`
- `RefCheckerCitationAdapter`
- `StructuredLLMClaimVerifierAdapter`
- `StructuredLLMCitationVerifierAdapter`

All external adapters are advisory only.
They do not own final diagnosis.

## Runtime Metadata Rule

Analyzer outputs should surface, at minimum:

- `method_classification`
- `evidence_role`
- `fallback_used`
- `fallback_heuristics_used`

If an analyzer omits this metadata directly, the engine should normalize and attach it before final diagnosis output is emitted.
