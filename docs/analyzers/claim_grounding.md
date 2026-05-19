# Claim Grounding Verification

GovRAG uses a safety-first approach to claim verification. By default, the system uses the `ConservativeEnsembleVerifier`, an uncalibrated but highly practical verification policy that guarantees a 0.0% false-pass rate on our core regression benchmarks.

## The Conservative Ensemble Policy

The `ConservativeEnsembleVerifier` runs two independent verification mechanisms in parallel:
1. **LLM Entailment (`LLMClaimEntailmentVerifierV1`)**: Provides high-recall semantic understanding to determine if the evidence supports the claim.
2. **Deterministic Heuristics (`HeuristicValueOverlapVerifier`)**: A strict, lexical-overlap and entity-matching engine that verifies facts (numbers, dates, entities) but has low semantic recall.

### Safety Gates

While the LLM acts as the primary reasoning engine, any claim that the LLM labels as `entailed` (supported) must pass three deterministic **Safety Gates**. If any gate triggers, the verdict is aggressively downgraded to `insufficient_evidence`.

1. **Heuristic Disagreement**: If the deterministic verifier explicitly marks the claim as `contradicted`, the LLM's `entailed` label is overridden. LLMs often rationalize away explicit contradictions; the heuristic engine does not.
2. **Missing Critical Fact Coverage**: If the extracted claim contains specific numbers or dates, those exact tokens *must* be present in the supporting evidence chunk. If the LLM claims support but the critical facts are missing, the claim is downgraded.
3. **Compound Claim Warning**: If the LLM identifies that a claim is compound (containing multiple independent facts) and cannot be atomicly verified, the claim is downgraded to force upstream decomposition.

### False-Pass as the Primary Risk Metric

In high-trust domains (e.g., policy, legal, government), a **False Pass** is the worst possible outcome. A false pass occurs when the system tells a user that a fabricated or contradicted claim is "supported by the evidence." 

The ensemble explicitly trades off some overall accuracy (sacrificing valid entailments that the heuristics are too rigid to confirm) to mathematically minimize the false-pass rate.

### Uncalibrated Outputs & RefChecker Divergence

**Important**: The `ConservativeEnsembleVerifier` is an *uncalibrated practical ensemble*. 
- It does not produce statistically calibrated confidence scores (unlike our future `ClaimCalibrationModel`).
- It diverges from academic implementations like RefChecker. We do not use triplet-extraction as our sole verification mechanism, as triplets often lose the nuanced context required for policy verification. Instead, we use whole-claim semantic entailment gated by lexical checks.

## Telemetry

To ensure observability, the pipeline outputs the following telemetry on every `ClaimEvidenceRecord` and `VerificationResult`:
- `verifier_policy`: The active policy (e.g., `conservative_ensemble`)
- `safety_gate_triggered`: Boolean indicating if an LLM override occurred
- `safety_gate_reason`: Text description of which gate triggered (e.g., `HEURISTIC_CONTRADICTED`)
- `verifier_disagreement`: Boolean indicating if the LLM and heuristic disagreed.
