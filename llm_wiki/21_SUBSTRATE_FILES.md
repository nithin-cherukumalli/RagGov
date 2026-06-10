# Substrate Files Reference

Complete reference of substrate implementation files in GovRAG.

## Grounding Substrate

Core files for claim extraction, evidence selection, and claim verification:

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/grounding/support.py` | `ClaimGroundingAnalyzer` — main entry point | CitationFaithfulnessAnalyzerV0, ClaimAwareSufficiencyAnalyzer, RetrievalDiagnosisAnalyzerV0, NCV, A2P, SemanticEntropy |
| `src/raggov/analyzers/grounding/claims.py` | Claim extraction logic | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/candidate_selection.py` | Evidence candidate chunk selection | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/evidence_layer.py` | Claim evidence record building | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/verifiers.py` | Multiple verifier implementations (heuristic, LLM, ensemble) | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/triplets.py` | Triplet extraction for structured verification | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/decomposer.py` | Compound claim decomposition | ClaimGroundingAnalyzer |
| `src/raggov/analyzers/grounding/diagnostic_rollups.py` | Claim diagnostic summary builder | ClaimGroundingAnalyzer, Layer6 |
| `src/raggov/analyzers/grounding/citation_faithfulness.py` | `CitationFaithfulnessProbe` implementation | CitationFaithfulnessAnalyzerV0 |
| `src/raggov/analyzers/grounding/critical_fact_normalizer.py` | Fact normalization utilities | Grounding subsystem |
| `src/raggov/analyzers/grounding/value_extraction.py` | Value extraction from claims | Grounding subsystem |

### Grounding Models

| File | Purpose |
|------|---------|
| `src/raggov/models/grounding.py` | `GroundingEvidenceBundle`, claim evidence records |

## Sufficiency Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/sufficiency/sufficiency.py` | `SufficiencyAnalyzer` — requirement-aware sufficiency | NCV, RetrievalDiagnosisAnalyzerV0 |
| `src/raggov/analyzers/sufficiency/claim_aware.py` | `ClaimAwareSufficiencyAnalyzer` — claim-tied sufficiency | NCV, RetrievalDiagnosisAnalyzerV0 |

## Retrieval Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/retrieval/scope.py` | `ScopeViolationAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval/stale.py` | `StaleRetrievalAnalyzer` | NCV, VersionValidity |
| `src/raggov/analyzers/retrieval/citation.py` | `CitationMismatchAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval/inconsistency.py` | `InconsistentChunksAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval/evidence_profile.py` | `RetrievalEvidenceProfilerV0` — builds evidence profile | NCV, multiple analyzers |
| `src/raggov/analyzers/retrieval/contradiction.py` | `ContradictionAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval/freshness.py` | `FreshnessAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval/relevance.py` | `RelevanceAnalyzer` | NCV |
| `src/raggov/analyzers/retrieval_diagnosis/retrieval_diagnosis.py` | `RetrievalDiagnosisAnalyzerV0` | NCV |

### Retrieval Models

| File | Purpose |
|------|---------|
| `src/raggov/models/retrieval_evidence.py` | `RetrievalEvidenceProfile`, chunk evidence roles |
| `src/raggov/models/retrieval_diagnosis.py` | `RetrievalDiagnosisReport` |

## Citation Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/citation_faithfulness/analyzer.py` | `CitationFaithfulnessAnalyzerV0` | NCV, decision policy |

### Citation Models

| File | Purpose |
|------|---------|
| `src/raggov/models/citation_faithfulness.py` | `CitationFaithfulnessReport`, citation support labels |

## Version Validity Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/version_validity/analyzer.py` | `TemporalSourceValidityAnalyzerV1`, `VersionValidityAnalyzerV1` | NCV |

### Version Validity Models

| File | Purpose |
|------|---------|
| `src/raggov/models/version_validity.py` | `VersionValidityReport`, document validity statuses |

## Security Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/security/injection.py` | `PromptInjectionAnalyzer` | NCV, decision policy |
| `src/raggov/analyzers/security/poisoning.py` | `PoisoningHeuristicAnalyzer` | NCV |
| `src/raggov/analyzers/security/privacy.py` | `PrivacyAnalyzer` | NCV |
| `src/raggov/analyzers/security/anomalies.py` | `RetrievalAnomalyAnalyzer` | NCV |

## Parsing Substrate

| File | Purpose | Downstream Consumers |
|------|---------|---------------------|
| `src/raggov/analyzers/parsing/parser_validation.py` | `ParserValidationAnalyzer` | NCV |

### Parser Validation Framework

| File | Purpose |
|------|---------|
| `src/raggov/parser_validation/engine.py` | Parser validation engine |
| `src/raggov/parser_validation/models.py` | Parser validation models |
| `src/raggov/parser_validation/profile.py` | Parser profile management |
| `src/raggov/parser_validation/validators/` | Validator implementations |

## Meta / Interpretation Substrate

### NCV (Node-wise Verification)

| File | Purpose |
|------|---------|
| `src/raggov/analyzers/verification/ncv.py` | `NCVPipelineVerifier` — node-wise evidence aggregation |
| `src/raggov/analyzers/verification/ncv_priority.py` | First failing node selection policy |
| `src/raggov/models/ncv.py` | `NCVReport`, `NCVNode`, `NCVNodeResult` |

### Layer6 Taxonomy Classification

| File | Purpose |
|------|---------|
| `src/raggov/analyzers/taxonomy_classifier/layer6.py` | `Layer6TaxonomyClassifier` — taxonomy mapping |

### A2P Attribution

| File | Purpose |
|------|---------|
| `src/raggov/analyzers/attribution/a2p.py` | `A2PAttributionAnalyzer` — counterfactual attribution |
| `src/raggov/analyzers/attribution/candidates.py` | Candidate cause generation (v2) |
| `src/raggov/analyzers/attribution/causal_chain.py` | Causal chain building |
| `src/raggov/analyzers/attribution/pinpoint_context.py` | Pinpoint finding extraction |
| `src/raggov/analyzers/attribution/scoring.py` | Attribution scoring |
| `src/raggov/analyzers/attribution/selection.py` | Primary/secondary cause selection |
| `src/raggov/analyzers/attribution/trace.py` | Attribution trace extraction |

### Semantic Entropy

| File | Purpose |
|------|---------|
| `src/raggov/analyzers/confidence/semantic_entropy.py` | `SemanticEntropyAnalyzer` — uncertainty detection |
| `src/raggov/analyzers/confidence/confidence.py` | Confidence calculation utilities |

## Shared Models

| File | Purpose |
|------|---------|
| `src/raggov/models/diagnosis.py` | `Diagnosis`, `AnalyzerResult`, `FailureType`, `FailureStage`, `ClaimResult`, `SufficiencyResult`, `ClaimAttribution`, `ClaimAttributionV2`, `CandidateCause` |
| `src/raggov/models/run.py` | `RAGRun` — runtime data container |
| `src/raggov/models/result_index.py` | `AnalyzerResultIndex` — result querying utility |
| `src/raggov/models/pinpoint.py` | `PinpointFinding`, `CausalChain`, `TrustDecision` |
| `src/raggov/models/chunk.py` | `RetrievedChunk` |
| `src/raggov/models/corpus.py` | Corpus models |
| `src/raggov/models/external_diagnosis.py` | `ExternalSignalDiagnosisProbe` |

## Orchestration

| File | Purpose | Classification |
|------|---------|----------------|
| `src/raggov/engine.py` | `DiagnosisEngine` — analyzer orchestration | Core runtime |
| `src/raggov/decision_policy.py` | `select_primary_failure_with_policy` — final failure selection | Danger zone |
| `src/raggov/decision_policy_support.py` | Evidence tier classification, candidate suppression | Danger zone |
| `src/raggov/external_signal_bridge.py` | External signal integration | External layer |
| `src/raggov/taxonomy.py` | `FAILURE_PRIORITY`, failure type definitions | Policy input |

## Base Classes

| File | Purpose |
|------|---------|
| `src/raggov/analyzers/base.py` | `BaseAnalyzer`, method classifications, evidence roles |
| `src/raggov/evaluators/base.py` | `ExternalEvaluationResult`, evaluator base classes |

## Calibration

| File | Purpose |
|------|---------|
| `src/raggov/calibration/core.py` | Core calibration logic |
| `src/raggov/calibration/claim_calibration.py` | Claim-level calibration |
