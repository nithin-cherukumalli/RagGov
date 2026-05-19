# Claim-Grounding Evaluation Report

- **Dataset**: `claim_grounding_100.jsonl`
- **Total cases**: 110
- **Verifier**: `heuristic`

## Summary Metrics

| Metric | Value |
|--------|-------|
| Overall accuracy | `59.1%` |
| **False-pass rate** | `0.0%` |
| False-fail rate | `0.0%` |
| Contradiction detection rate | `75.0%` |
| Evidence chunk recall | `77.8%` |
| Safety gate precision | `0.0%` |
| Fallback rate | `0.0%` |

## Per-Label Metrics

| Label | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| entailed | 1.000 | 0.778 | 0.875 | 35 | 0 | 10 |
| unsupported | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |
| contradicted | 0.857 | 0.750 | 0.800 | 30 | 5 | 10 |

## Raw Counts

- Correct predictions: **65** / 110
- False passes (bad answers let through): **0**
- False fails (good answers rejected): **0**
- Contradictions correctly caught: **30**
- Predictions using fallback: **0**

## Ensemble & Disagreement Metrics

- Safety gate downgrades: **0**
- LLM Supported vs Heuristic Contradicted: **0**

> **Note**: False-pass rate is the primary risk metric for high-trust RAG.
> A false pass means a fabricated or contradicted claim was silently accepted.

## Domain-Wise Breakdown

| Domain | Cases | Accuracy | False-Pass Rate |
|--------|-------|----------|----------------|
| education | 7 | `71.4%` | `0.0%` |
| finance | 18 | `55.6%` | `0.0%` |
| government | 22 | `59.1%` | `0.0%` |
| healthcare | 21 | `57.1%` | `0.0%` |
| product_manuals | 16 | `56.2%` | `0.0%` |
| science | 12 | `58.3%` | `0.0%` |
| software | 14 | `64.3%` | `0.0%` |
