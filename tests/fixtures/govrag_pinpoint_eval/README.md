# GovRAG Pinpoint Eval Gold Sets

This directory contains a tiny offline gold set for evaluating structured:

- `first_failing_node`
- `pinpoint_findings`
- `causal_chains`
- non-gating honesty fields

Two benchmark files live here:

- `pinpoint_eval_gold_v1.json`
  - 8-case smoke benchmark
  - useful for fast regression checks and mismatch repair work
- `pinpoint_eval_gold_v2_30.json`
  - 30-case generalization benchmark
  - still offline and deterministic, but broad enough to expose node/root-cause drift

Format:

```json
{
  "evaluation_status": "pinpoint_eval_gold_v1_small_unvalidated",
  "cases": [
    {
      "case_id": "retrieval_miss",
      "description": "Short description",
      "run_fixture": "fixtures/insufficient_context.json",
      "engine_config": {
        "mode": "native",
        "enable_ncv": true,
        "enable_a2p": true,
        "use_llm": false
      },
      "expected": {
        "primary_failure": "INSUFFICIENT_CONTEXT",
        "first_failing_node": "retrieval_coverage",
        "pinpoint_node": "retrieval_coverage",
        "root_cause": "retrieval_coverage_gap",
        "fix_category": "retrieval_recall",
        "secondary_nodes": [],
        "affected_claim_ids": [],
        "affected_chunk_ids": [],
        "affected_doc_ids": [],
        "human_review_required": true,
        "recommended_for_gating": false,
        "calibration_status": "uncalibrated"
      }
    }
  ]
}
```

For the v2 30-case set, every case must include:

- `primary_failure`
- `first_failing_node`
- `pinpoint_node`
- `root_cause`
- `fix_category`
- `secondary_nodes`
- `affected_claim_ids`
- `affected_doc_ids`
- `human_review_required`
- `recommended_for_gating`
- `calibration_status`

`run_fixture` may point either to:

- a native `RAGRun` JSON fixture
- a fixture with `query`, `retrieved_chunks`, `final_answer`, and optional
  corpus/metadata fields that can be converted into a `RAGRun`

The harness must run offline:

- no real LLM
- no API keys
- no network
- no model download
