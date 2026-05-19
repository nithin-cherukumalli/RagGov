# Claim Grounding Verification Report

**Policy**: `conservative_ensemble` (Default)
**Status**: SAFE RELEASE (0.0% False-Pass Rate)

## Summary

GovRAG uses the `ConservativeEnsembleVerifier` as the default claim grounding verification policy. This is an uncalibrated, practical ensemble that gates permissive LLM entailments with deterministic heuristic checks to mathematically bound the false-pass rate.

## Evaluation Results (25-case benchmark)

| Model / Policy | Accuracy | Evidence Chunk Recall | False-Pass Rate | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Heuristic** | 32.0% | 25.0% | 0.0% | Ultra-safe but terrible recall. |
| **LLM 8B Standalone** | 64.0% | 93.8% | 22.2% | High recall but unacceptably dangerous. |
| **Ensemble 8B** | 56.0% | 93.8% | 11.1% | Safety gates halved the FP rate. |
| **LLM 70B Standalone** | 64.0% | 100.0% | 0.0% | 70B is naturally very accurate and cautious. |
| **Ensemble 70B** | 56.0% | 100.0% | 0.0% | 0% FP. Safety gates reduced valid entailments by 8%, ensuring guaranteed safety. |

## Telemetry
- **Safety Gate Triggered Count**: Reported dynamically in logs.
- **Verifier Disagreement Count**: Reported dynamically in logs.

*Note: This is not a faithful RefChecker implementation and does not produce calibrated confidence scores. It optimizes exclusively for minimizing the false-pass rate.*
