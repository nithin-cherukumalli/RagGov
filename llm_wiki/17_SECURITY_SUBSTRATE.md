# Security Substrate

Core files:

- `src/raggov/analyzers/security/injection.py`
- `src/raggov/analyzers/security/poisoning.py`
- `src/raggov/analyzers/security/privacy.py`
- `src/raggov/analyzers/security/anomalies.py`

## Evidence Producers

- prompt injection detection
- poisoning heuristics
- privacy request detection
- retrieval anomaly detection

## Honest Classification

- `PromptInjectionAnalyzer`: `practical_approximation`
- `PoisoningHeuristicAnalyzer`: `heuristic_baseline`
- `PrivacyAnalyzer`: `heuristic_baseline`
- `RetrievalAnomalyAnalyzer`: `heuristic_baseline`

## Rule

Security analyzers produce blocking evidence or risk signals.
Meta layers may interpret them but may not dilute or invent them.
