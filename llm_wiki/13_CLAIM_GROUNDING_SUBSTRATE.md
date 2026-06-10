# Claim Grounding Substrate

## Why This Is The Priority

Claim grounding is currently the highest-leverage substrate to harden first.

Core files:

- `src/raggov/analyzers/grounding/support.py`
- `src/raggov/analyzers/grounding/evidence_layer.py`
- `src/raggov/analyzers/grounding/claims.py`
- `src/raggov/analyzers/grounding/candidate_selection.py`
- `src/raggov/analyzers/grounding/verifiers.py`
- `src/raggov/models/grounding.py`

## What It Produces

The grounding substrate produces:

- extracted claims
- candidate evidence chunks
- claim-level labels
- supporting / contradicting / neutral chunk ids
- verifier method metadata
- fallback usage metadata
- grounding evidence bundles

## Real Dependencies On It

Direct or near-direct downstream consumers include:

- `CitationFaithfulnessAnalyzerV0`
- `CitationFaithfulnessProbe`
- `ClaimAwareSufficiencyAnalyzer`
- `RetrievalDiagnosisAnalyzerV0`
- `SemanticEntropyAnalyzer`
- `A2PAttributionAnalyzer`
- final diagnosis assembly in `engine.py`

## Current Risk

`ClaimGroundingAnalyzer` supports multiple verifier modes and fallback paths.
That flexibility is useful, but it also means trust metadata must stay explicit.
LLM-backed grounding configuration must behave consistently across `llm_client` and `llm_fn` paths.

## Hardening Goals

1. Make fallback provenance impossible to miss.
2. Improve claim extraction honesty and skip semantics.
3. Improve evidence candidate quality before policy tweaks.
4. Keep calibration claims conservative.
5. Keep LLM-backed verifier mode semantics consistent across supported runtime interfaces.
