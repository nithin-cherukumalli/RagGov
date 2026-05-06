# GovRAG Evidence Layer Fixtures

This directory holds JSON fixtures for the evidence-layer reliability gate.

Current scope:
- 12 initial cases covering clean runs, retrieval failures, citation failures,
  grounding failures, stale/version failures, parser failures, security failures,
  degraded external mode, and no-claim clean paths.

Design goals:
- offline and deterministic
- directly convertible into `RAGRun`
- stable enough for mutation-style tests
- scalable to 30 cases without changing loader code

Fixture schema:

```json
{
  "case_id": "string",
  "description": "string",
  "query": "string",
  "retrieved_chunks": [],
  "final_answer": "string",
  "citations": [],
  "cited_doc_ids": [],
  "corpus_metadata": {},
  "parser_validation_profile": null,
  "mode": "native or external-enhanced",
  "expected": {
    "primary_failure": "optional string",
    "not_clean": true,
    "required_reports": [],
    "required_evidence_signals": [],
    "expected_missing_external_providers": [],
    "expected_degraded_external_mode": false
  }
}
```

Local test-only extensions may also appear:
- `answer_confidence`
- `metadata`
- `enabled_external_providers`
- `retrieval_relevance_provider`
- `mock_external_results`

These are consumed only by the evidence-layer stress tests.
