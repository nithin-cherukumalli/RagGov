# GovRAG Limitations

GovRAG provides stage-aware failure attribution, but the diagnostic strength depends on the availability of signals.

## Diagnosis Modes

GovRAG operates under three distinct modes:
- **external-enhanced**: This is the default. The engine will utilize LLM-based verification, adapter integrations (e.g. RAGAS, DeepEval), and other semantic features if available. When dependencies are missing, the system warns the user, gracefully falling back to native heuristics, and lists unfulfilled features in `missing_external_providers`.
- **native**: GovRAG will forcefully disable any external calls (no LLM, no adapters). Grounding defaults to heuristics. This mode is completely offline and fast but may have a higher false-negative rate for complex semantic contradictions.
- **calibrated**: (Unimplemented) Future mode for ARES PPI-corrected outputs.

## Grounding Limits

In `native` mode, the `heuristic_claim_verifier` detects simple token overlaps but will struggle with:
- Heavy paraphrasing
- Semantic contradictions that use identical entities
- Implicit claims

For production use cases requiring high-precision grounding, users should configure `external-enhanced` mode with an active `llm_client`.

## Security Detection

Native security detection (`PromptInjectionAnalyzer`, `PoisoningHeuristicAnalyzer`) uses heuristics and known adversarial patterns. It is not a replacement for a dedicated LLM firewall.
