# Real Failure Evaluation Protocol

This document defines how GovRAG should earn credibility on real-world RAG failures.
It is a protocol, not a completed validation report.

## Goal

Publish an honest native-mode accuracy report on 50 real RAG failures.

The report must separate:

- correct diagnosis
- acceptable adjacent diagnosis
- wrong diagnosis

## Preferred Data Sources

1. anonymized partner or user production failures
2. public benchmark failures with reproducible context
3. manually curated public reproductions with documented provenance

## Required Labels Per Case

- `case_id`
- `source_type`
- `query`
- `retrieved_context`
- `final_answer`
- `expected_primary_failure`
- `expected_root_stage`
- `acceptable_alternatives`
- `notes`

## Evaluation Rules

- Evaluate `native` mode first.
- External-enhanced results may appear only as a secondary comparison.
- If a case is ambiguous, document the ambiguity instead of forcing a win.
- If GovRAG misses the case, keep the miss in the report and add it to the benchmark intake queue.

## Publishable Outputs

- total accuracy
- per-failure-type accuracy
- confusion patterns
- top misses and why they happened
- which misses are substrate failures vs. policy failures

## Anti-Hype Rule

Do not describe this report as scientific validation.
It is evidence of honest progress toward a reputable diagnosis tool.

