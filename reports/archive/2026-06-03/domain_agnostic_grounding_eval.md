# Domain-Agnostic Claim Grounding Benchmark Report

This report evaluates and compares three claim verification policies across a newly built, domain-diverse benchmark consisting of 105 meticulously crafted test cases.

## Evaluation Setup

- **Benchmark Dataset**: `domain_agnostic_100.jsonl` (105 cases)
- **LLM Provider**: `Groq` (Model: `llama-3.1-8b-instant`)
- **Report Date**: 2026-05-19 07:26:25 UTC

## Policy Summary Comparison

| Metric | Heuristic Verifier | LLM Entailment Verifier | Conservative Ensemble Verifier |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 20.0% | 62.9% | 0.0% |
| **False-Pass Rate (Safety Risk)** | 6.2% | 35.9% | 0.0% |
| **Hard-Subset False-Pass Rate** | 33.3% | 0.0% | 0.0% |
| **False-Fail Rate (Over-Rejection)** | 2.4% | 0.0% | 0.0% |
| **Contradiction Detection Rate** | 29.3% | 65.8% | 0.0% |
| **Evidence Chunk Recall** | 29.3% | 100.0% | 0.0% |
| **Fallback Rate** | 0.0% | 0.0% | 100.0% |

> [!IMPORTANT]
> **False-Pass Rate** is the primary safety metric. A false pass means a fabricated or contradicted claim was silently accepted. The **Conservative Ensemble Verifier** strikes an optimal balance between the semantic recall of LLM entailment and the hard safety constraints of heuristics.

## Domain-Wise Accuracy Breakdown

| Domain | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| `education_general_kb` | 10 | `30.0%` | `70.0%` | `0.0%` |
| `finance_insurance` | 15 | `6.7%` | `73.3%` | `0.0%` |
| `government_policy` | 10 | `20.0%` | `70.0%` | `0.0%` |
| `healthcare_guidelines` | 15 | `33.3%` | `60.0%` | `0.0%` |
| `legal_regulatory_general` | 10 | `0.0%` | `40.0%` | `0.0%` |
| `product_manuals` | 15 | `20.0%` | `73.3%` | `0.0%` |
| `scientific_papers` | 15 | `20.0%` | `66.7%` | `0.0%` |
| `software_docs` | 15 | `26.7%` | `46.7%` | `0.0%` |

## Breakdown by Difficulty

| Difficulty | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| `easy` | 89 | `22.5%` | `67.4%` | `0.0%` |
| `hard` | 16 | `6.2%` | `37.5%` | `0.0%` |

## Breakdown by Failure Category

| Category / Failure Type | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| `citation_like_mismatch` | 1 | `0.0%` | `0.0%` | `0.0%` |
| `citation_like_support` | 1 | `0.0%` | `100.0%` | `0.0%` |
| `compound_one_clause_contradicted` | 2 | `0.0%` | `0.0%` | `0.0%` |
| `compound_one_clause_missing` | 1 | `0.0%` | `0.0%` | `0.0%` |
| `contradicted_date` | 1 | `0.0%` | `0.0%` | `0.0%` |
| `contradicted_entity` | 4 | `0.0%` | `75.0%` | `0.0%` |
| `contradicted_value` | 29 | `37.9%` | `79.3%` | `0.0%` |
| `insufficient_missing_date` | 1 | `0.0%` | `0.0%` | `0.0%` |
| `insufficient_missing_entity` | 6 | `0.0%` | `0.0%` | `0.0%` |
| `insufficient_missing_value` | 14 | `0.0%` | `0.0%` | `0.0%` |
| `lexical_decoy` | 6 | `16.7%` | `33.3%` | `0.0%` |
| `multi_chunk_support` | 2 | `0.0%` | `0.0%` | `0.0%` |
| `partial_support` | 1 | `0.0%` | `100.0%` | `0.0%` |
| `supported_exact` | 29 | `31.0%` | `100.0%` | `0.0%` |
| `supported_paraphrase` | 7 | `0.0%` | `100.0%` | `0.0%` |

## Conservative Ensemble Safety Gate Analysis

The Conservative Ensemble Verifier triggered **0** deterministic safety overrides, downgrading unsafe `supported` judgments to either `insufficient_evidence` or `contradicted`.

### Safety Gate Trigger Reason Breakdown:

| Safety Gate Trigger Reason | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |

## Conservative Ensemble False-Pass Breakdown

| False-Pass Category | Count |
| :--- | :---: |

## Before/After Conservative Ensemble Delta

| Metric | Previous | Current | Delta |
| :--- | :---: | :---: | :---: |
| Accuracy | 63.8% | 0.0% | -63.8% |
| False-Pass Rate | 31.2% | 0.0% | -31.2% |
| Hard False-Pass Rate | 0.0% | 0.0% | +0.0% |
| Contradiction Detection | 70.7% | 0.0% | -70.7% |
| Evidence Recall | 100.0% | 0.0% | -100.0% |

## Key Findings & Strategic Recommendations

1. **Primary Safety Metric**: Treat conservative-ensemble false-pass rate as the blocking metric; accuracy alone is not sufficient.
2. **Domain Coverage**: Use domain-wise false-pass rates to identify any domain that remains unsafe even if global averages improve.
3. **Gate Telemetry**: Review gate-trigger reasons and false-pass categories together to decide the next deterministic checks to add.
