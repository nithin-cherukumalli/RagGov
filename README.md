# RagGov

Diagnose why your RAG answer failed, whether the evidence was unsafe, and whether the system should have answered at all.

## What it does

RagGov takes a `RAGRun` object containing a query, retrieved chunks, citations, final answer, and optional corpus metadata. It runs deterministic analyzers for retrieval quality, context sufficiency, grounding, security, and confidence. It returns a structured `Diagnosis` with a primary failure, supporting evidence, security risk, abstention decision, and recommended fix.

## Why it exists

RAGAS, DeepEval, and LangSmith are useful when you need scoring, tracing, and evaluation workflows. RagGov focuses on the next question engineers ask after a score drops: what failed, where did it fail, and should the system have answered at all? It is built to produce an inspectable diagnosis rather than another black-box aggregate metric.

## Quick Install

```bash
pip install raggov
```

## Quick Start

### CLI

```bash
raggov diagnose run.json
```

Sample output:

```text
+--------------------------------------------------------------------------------+
| Diagnosis: 9ad6b53c-cc9a-4d9d-b0e0-7157fd4c4c25                                 |
+--------------------------------------------------------------------------------+
| Run ID                 9ad6b53c-cc9a-4d9d-b0e0-7157fd4c4c25                       |
| Timestamp              2026-04-10T11:22:04.118000+00:00                          |
| Primary failure        PROMPT_INJECTION                                          |
| Should have answered   No                                                        |
| Security risk          HIGH                                                      |
| Confidence             0.29                                                      |
| Evidence               - prompt-injection-chunk-2: 5 hit(s): ignore previous...   |
| Recommended fix        Retrieved chunk(s) contain instruction-like content        |
|                        consistent with prompt injection. Sanitize corpus or add   |
|                        a pre-retrieval content filter.                           |
| Checks                 StaleRetrievalAnalyzer       pass                          |
|                        CitationMismatchAnalyzer     pass                          |
|                        PromptInjectionAnalyzer      fail  PROMPT_INJECTION        |
|                        ConfidenceAnalyzer           fail  LOW_CONFIDENCE          |
+--------------------------------------------------------------------------------+
Wrote raw diagnosis JSON to ./9ad6b53c-cc9a-4d9d-b0e0-7157fd4c4c25_diagnosis.json
```

### Python SDK

```python
from raggov import RAGRun, RetrievedChunk, diagnose

run = RAGRun(
    query="Summarize the refund rules.",
    retrieved_chunks=[
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Refunds are available within fourteen days.",
            source_doc_id="doc-1",
            score=0.91,
        )
    ],
    final_answer="Refunds are available within fourteen days.",
)
diagnosis = diagnose(run)
print(diagnosis.primary_failure, diagnosis.should_have_answered)
```

### Sample Diagnosis JSON

Prompt injection diagnosis:

```json
{
  "run_id": "9ad6b53c-cc9a-4d9d-b0e0-7157fd4c4c25",
  "primary_failure": "PROMPT_INJECTION",
  "secondary_failures": [
    "UNSUPPORTED_CLAIM",
    "SCOPE_VIOLATION",
    "INCONSISTENT_CHUNKS",
    "LOW_CONFIDENCE"
  ],
  "root_cause_stage": "SECURITY",
  "should_have_answered": false,
  "security_risk": "HIGH",
  "confidence": 0.29,
  "claim_results": [],
  "evidence": [
    "prompt-injection-chunk-2: 5 hit(s): ignore (all |previous |above |prior )?(instructions?|prompts?|context|rules?); you are now; system prompt; reveal (your |the )?(system|prompt|instruction); repeat (everything|all|the prompt)",
    "prompt-injection-chunk-2 overlap=0.00",
    "{\"claim_text\":\"Helio subscribers can request a refund within fourteen days of the initial purchase, and annual renewals are refundable only if premium features were not used after renewal.\",\"label\":\"contradicted\",\"supporting_chunk_ids\":[\"prompt-injection-chunk-1\"],\"confidence\":0.8181818181818182}",
    "base score: 1.00",
    "blended caller answer_confidence 0.77: score 0.89",
    "prior result PromptInjectionAnalyzer status fail: -0.20",
    "final score: 0.29"
  ],
  "recommended_fix": "Retrieved chunk(s) contain instruction-like content consistent with prompt injection. Sanitize corpus or add a pre-retrieval content filter.",
  "checks_run": [
    "StaleRetrievalAnalyzer",
    "CitationMismatchAnalyzer",
    "InconsistentChunksAnalyzer",
    "ScopeViolationAnalyzer",
    "SufficiencyAnalyzer",
    "ClaimGroundingAnalyzer",
    "PromptInjectionAnalyzer",
    "RetrievalAnomalyAnalyzer",
    "PoisoningHeuristicAnalyzer",
    "ConfidenceAnalyzer"
  ],
  "checks_skipped": [],
  "analyzer_results": [
    {
      "analyzer_name": "PromptInjectionAnalyzer",
      "status": "fail",
      "failure_type": "PROMPT_INJECTION",
      "stage": "SECURITY",
      "score": null,
      "security_risk": "HIGH",
      "evidence": [
        "prompt-injection-chunk-2: 5 hit(s): ignore (all |previous |above |prior )?(instructions?|prompts?|context|rules?); you are now; system prompt; reveal (your |the )?(system|prompt|instruction); repeat (everything|all|the prompt)"
      ],
      "remediation": "Retrieved chunk(s) contain instruction-like content consistent with prompt injection. Sanitize corpus or add a pre-retrieval content filter."
    },
    {
      "analyzer_name": "ConfidenceAnalyzer",
      "status": "fail",
      "failure_type": "LOW_CONFIDENCE",
      "stage": "CONFIDENCE",
      "score": 0.29,
      "security_risk": null,
      "evidence": [
        "base score: 1.00",
        "blended caller answer_confidence 0.77: score 0.89",
        "prior result PromptInjectionAnalyzer status fail: -0.20",
        "final score: 0.29"
      ],
      "remediation": "Confidence too low to trust output. Consider abstaining, re-retrieving, or requesting human review."
    }
  ],
  "created_at": "2026-04-10T11:22:04.118000Z"
}
```

## What it checks

| Analyzer | Description |
| --- | --- |
| `StaleRetrievalAnalyzer` | Detects retrieved documents older than a configurable freshness threshold. |
| `CitationMismatchAnalyzer` | Detects answer citations that were not present in retrieved context. |
| `InconsistentChunksAnalyzer` | Flags simple contradiction signals across retrieved chunks. |
| `ScopeViolationAnalyzer` | Flags chunks that have weak keyword overlap with the query. |
| `SufficiencyAnalyzer` | Checks whether retrieved context covers enough query terms to answer. |
| `ClaimGroundingAnalyzer` | Extracts answer claims and checks support against retrieved chunks. |
| `PromptInjectionAnalyzer` | Detects instruction-like and exfiltration text in retrieved chunks. |
| `RetrievalAnomalyAnalyzer` | Detects score cliffs, outliers, and near-duplicate chunks. |
| `PoisoningHeuristicAnalyzer` | Detects high-score answer-steering chunks. |
| `ConfidenceAnalyzer` | Aggregates answer confidence, analyzer results, and retrieval scores. |

## Failure Taxonomy

- `STALE_RETRIEVAL`: Retrieved documents are outdated and may not reflect current information.
- `SCOPE_VIOLATION`: Retrieved documents appear off-topic for the user's query.
- `CITATION_MISMATCH`: The answer cites sources that were not present in the retrieved context.
- `INCONSISTENT_CHUNKS`: Retrieved chunks contain potentially inconsistent or conflicting information.
- `INSUFFICIENT_CONTEXT`: Retrieved context does not contain enough information to answer reliably.
- `UNSUPPORTED_CLAIM`: The answer contains claims that are not supported by retrieved evidence.
- `CONTRADICTED_CLAIM`: The answer contains claims contradicted by retrieved evidence.
- `PROMPT_INJECTION`: Retrieved content contains instruction-like text consistent with prompt injection.
- `SUSPICIOUS_CHUNK`: A retrieved chunk shows signs of answer-steering or corpus poisoning.
- `RETRIEVAL_ANOMALY`: Retrieval results show statistical anomalies that may indicate manipulation.
- `LOW_CONFIDENCE`: Available confidence signals indicate the output may not be trustworthy.
- `CLEAN`: No diagnostic failure was detected.

## Architecture

```text
src/raggov/
├── models/        # Pydantic schemas for runs, chunks, corpus entries, and diagnoses
├── analyzers/     # Deterministic checks for retrieval, sufficiency, grounding, security, and confidence
├── connectors/    # Input connectors for external run formats
├── io/            # JSON serialization and append-only audit logging
├── plugins/       # Extension interfaces and registry
├── cli.py         # Typer/Rich command-line interface
├── engine.py      # Analyzer orchestration and diagnosis merging
└── taxonomy.py    # Failure vocabulary, stage descriptions, priorities, and default remediations
```

## Roadmap

- `v1.0`: Deterministic diagnostic core, CLI, SDK, fixtures, audit JSONL, and baseline analyzers.
- `v1.1`: Plugin registration, richer connector support, and configurable analyzer suites.
- `v1.5`: Semantic NLI grounding, semantic inconsistency checks, and optional LLM judge improvements.
- `v2`: Calibrated confidence, semantic entropy, conformal-style abstention, and production trace integrations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
