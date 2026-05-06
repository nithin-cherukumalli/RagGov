# GovRAG

**Diagnosis for production RAG systems.**

GovRAG tells you why a RAG answer failed, where the failure originated, whether the evidence was unsafe, and whether the system should have answered at all.

Most RAG tooling measures answer quality or traces execution. GovRAG is built for a different job: failure attribution.

## Overview

Production RAG systems do not fail in one place.

They fail because table structure was flattened during parsing, because chunk boundaries broke a governing clause, because embeddings collapsed near-duplicate records, because retrieval missed the controlling evidence, because the model ignored available context, or because unsafe content entered the context window and steered generation.

Those are different engineering failures. They should not be collapsed into a single "hallucination" bucket.

GovRAG sits on top of an existing RAG pipeline, accepts a normalized `RAGRun`, executes a diagnosis suite, and returns a structured `Diagnosis` that explains:

- the primary failure
- the root-cause stage
- the secondary failures
- whether the system should have answered
- the security risk
- the confidence level
- the evidence supporting the diagnosis
- the recommended next fix

This is not a score wrapper. It is a diagnosis layer for systems that need to be debugged, governed, and improved deliberately.

## Why GovRAG

RAGAS, DeepEval, and similar frameworks are useful when the question is:

> Was the answer good?

Tracing and observability platforms such as LangSmith are useful when the question is:

> What happened during execution?

GovRAG is built for the next question:

> What failed first, what followed from it, and what should the engineer fix?

That distinction matters in practice. The same visibly bad answer can come from fundamentally different causes:

- retrieval returned the wrong near-duplicate document
- chunking separated the legal condition from the operative clause
- the parser destroyed a table before retrieval ever ran
- the model fabricated a detail despite relevant evidence being present
- the retrieved context itself contained prompt-injection content

Good engineering decisions depend on telling those apart.

## What Is Novel

GovRAG is built around an explicit failure model for RAG systems rather than a generic output-quality abstraction.

### Layer 6-style Failure Taxonomy

GovRAG uses a stage-aware taxonomy aligned with how complex RAG pipelines actually break:

- parsing
- chunking
- embedding
- retrieval
- grounding
- sufficiency
- security
- confidence

The value of that taxonomy is practical. Downstream symptoms often obscure upstream causes. An unsupported claim may be a retrieval problem, a chunk-boundary problem, a parser problem, or a generation problem. GovRAG is designed to separate those cases.

### A2P-style Attribution

GovRAG is designed to move from symptom detection to attribution:

- what failed
- what likely caused it
- what downstream issues were side effects
- what action is most likely to improve the system

The goal is not to stop at:

> unsupported claim

The goal is to say:

> unsupported claim because retrieval missed the governing clause, likely due to chunk boundary loss, and the next intervention should be clause-preserving chunking or broader retrieval.

That is the difference between an evaluation artifact and an engineering tool.

### Entropy-based Confabulation Signals

GovRAG’s direction includes uncertainty signals based on semantic instability, not just answer fluency.

If repeated answers over the same evidence converge semantically, confidence should rise. If they diverge semantically, the system is confabulating even when each answer looks polished in isolation.

That is a stronger production signal than surface-level confidence heuristics.

## What GovRAG Checks

GovRAG currently ships with deterministic analyzers across retrieval, sufficiency, grounding, security, and confidence.

| Domain | Analyzer | Purpose |
| --- | --- | --- |
| Retrieval | `StaleRetrievalAnalyzer` | Detects outdated retrieved documents |
| Retrieval | `CitationMismatchAnalyzer` | Detects citations outside the retrieved context window |
| Retrieval | `InconsistentChunksAnalyzer` | Flags contradiction-like signals across retrieved chunks |
| Retrieval | `ScopeViolationAnalyzer` | Detects off-topic retrieval relative to the query |
| Sufficiency | `SufficiencyAnalyzer` | Determines whether the retrieved context is sufficient to answer reliably |
| Grounding | `ClaimGroundingAnalyzer` | Checks whether answer claims are entailed, unsupported, or contradicted |
| Security | `PromptInjectionAnalyzer` | Detects instruction-like, exfiltration, and jailbreak-style content in retrieved chunks |
| Security | `RetrievalAnomalyAnalyzer` | Flags anomalous retrieval patterns consistent with manipulation or poisoning |
| Security | `PoisoningHeuristicAnalyzer` | Detects answer-steering chunks with suspiciously strong retrieval characteristics |
| Confidence | `ConfidenceAnalyzer` | Aggregates caller confidence, retrieval quality, and analyzer outcomes into a final trust signal |

## Failure Taxonomy

GovRAG uses typed failure classes so diagnoses are stable, inspectable, and automatable.

| Failure Type | Meaning |
| --- | --- |
| `STALE_RETRIEVAL` | Retrieved material is outdated relative to the task |
| `SCOPE_VIOLATION` | Retrieved context is likely off-topic |
| `CITATION_MISMATCH` | The answer cites sources outside the retrieved context |
| `INCONSISTENT_CHUNKS` | Retrieved chunks show contradiction-like signals |
| `INSUFFICIENT_CONTEXT` | The retrieved context does not contain enough information to answer reliably |
| `UNSUPPORTED_CLAIM` | The answer makes claims not supported by retrieved evidence |
| `CONTRADICTED_CLAIM` | The answer conflicts with retrieved evidence |
| `PROMPT_INJECTION` | Retrieved content contains instruction-like or adversarial prompt material |
| `SUSPICIOUS_CHUNK` | A chunk exhibits poisoning-like answer-steering behavior |
| `RETRIEVAL_ANOMALY` | Retrieval behavior shows statistical or structural anomalies |
| `LOW_CONFIDENCE` | Aggregate signals indicate the output is not trustworthy |
| `CLEAN` | No significant failure was detected |

Every diagnosis also includes:

- `root_cause_stage`
- `should_have_answered`
- `security_risk`
- `confidence`
- `evidence`
- `recommended_fix`

## Install

```bash
pip install raggov
```

For external-enhanced installs without LLM providers:

```bash
pip install "raggov[external]"
```

For local development:

```bash
pip install -e ".[external,llm]"
```

## Quick Start

### Python SDK

```python
from raggov import RAGRun, RetrievedChunk, diagnose

run = RAGRun(
    query="What is the refund window?",
    retrieved_chunks=[
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Refunds are available within fourteen days of purchase.",
            source_doc_id="doc-1",
            score=0.92,
        )
    ],
    final_answer="Refunds are available within fourteen days of purchase.",
    cited_doc_ids=["doc-1"],
)

diagnosis = diagnose(run)

print(diagnosis.summary())
print(diagnosis.primary_failure)
print(diagnosis.should_have_answered)
```

### CLI

```bash
raggov diagnose run.json --mode external-enhanced
```

GovRAG supports the following diagnosis modes:
- `external-enhanced` (default): Uses package-based external signals that install locally (`ragas`, `deepeval`, `refchecker`, `ragchecker`) and keeps heavyweight or credentialed paths like cross-encoder retrieval, structured LLM verification, and A2P as explicit opt-ins.
- `native`: Uses only native heuristics. Faster and operates entirely offline with no external dependencies.
- `calibrated`: (Reserved for future ARES PPI-corrected outputs)

GovRAG will:

1. validate the run input
2. execute the analyzer suite
3. print a structured diagnosis panel
4. write raw diagnosis JSON to the working directory

### Semantic Entropy Configuration

`SemanticEntropyAnalyzer` supports both deterministic claim-label entropy and LLM sampling mode.
When LLM sampling is enabled, semantic clustering can use either:

- LLM-based NLI via `llm_fn` with no extra dependencies
- exact local NLI clustering via `sentence-transformers` and a configured `nli_model`

```python
from raggov.engine import DiagnosisEngine

engine = DiagnosisEngine(
    config={
        "use_llm": True,
        "llm_fn": my_llm_fn,
        "n_samples": 5,
        "temperature": 0.7,
        # Optional local NLI model for exact Farquhar-style clustering:
        "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
    }
)
```

If `nli_model` is not configured, GovRAG uses `llm_fn` as the semantic equivalence judge in LLM mode.

## Example Diagnosis

```json
{
  "run_id": "9ad6b53c-cc9a-4d9d-b0e0-7157fd4c4c25",
  "primary_failure": "PROMPT_INJECTION",
  "secondary_failures": [
    "UNSUPPORTED_CLAIM",
    "LOW_CONFIDENCE"
  ],
  "root_cause_stage": "SECURITY",
  "should_have_answered": false,
  "security_risk": "HIGH",
  "confidence": 0.29,
  "evidence": [
    "prompt-injection-chunk-2: instruction-like content detected",
    "prior result PromptInjectionAnalyzer status fail: -0.20",
    "final score: 0.29"
  ],
  "recommended_fix": "Retrieved chunk(s) contain instruction-like content consistent with prompt injection. Sanitize corpus or add a pre-retrieval content filter."
}
```

The important point is not that the answer was bad. The important point is that the output identifies the mechanism of failure, the stage where it originated, and the next engineering action.

## Core Data Model

GovRAG is built around a small public contract so it can sit on top of existing RAG systems without forcing a framework rewrite.

### `RAGRun`

Represents one end-to-end RAG execution:

- query
- retrieved chunks
- final answer
- cited document ids
- optional answer confidence
- optional trace
- optional corpus entries

### `Diagnosis`

Represents GovRAG’s output:

- primary failure
- secondary failures
- root-cause stage
- abstention decision
- security risk
- confidence
- evidence
- recommended fix
- analyzer results

## Public API

GovRAG exposes a deliberately small top-level SDK:

```python
from raggov import (
    diagnose,
    diagnose_dict,
    diagnose_file,
    RAGRun,
    RetrievedChunk,
    CorpusEntry,
    Diagnosis,
)
```

## Architecture

```text
src/raggov/
├── analyzers/     # Failure detectors across retrieval, sufficiency, grounding, security, and confidence
├── models/        # Pydantic models for runs, chunks, corpus entries, and diagnoses
├── io/            # Serialization and audit logging
├── connectors/    # Input adapters
├── plugins/       # Extension interfaces and registry
├── engine.py      # Analyzer orchestration and diagnosis synthesis
├── taxonomy.py    # Failure vocabulary, priority ordering, and default remediations
└── cli.py         # Command-line interface
```

## Intended Use

GovRAG is a strong fit when:

- answer correctness alone is not enough
- you need failure localization rather than only scoring
- abstention behavior matters
- retrieved evidence can itself be unsafe
- your team needs machine-readable diagnoses for CI, audits, or automated triage
- debugging time matters and generic "hallucination" labels are no longer useful

It is particularly well suited for:

- public-sector and enterprise RAG
- legal, policy, and regulatory retrieval systems
- internal knowledge systems with compliance requirements
- high-stakes search-and-answer products
- multi-stage RAG pipelines where reliability work needs to be prioritized

## Positioning

If answer-quality frameworks ask:

> Was the answer good?

and tracing tools ask:

> What happened during execution?

GovRAG asks:

> What failed, where did it fail, and what should the engineer fix first?

That is the category.

GovRAG is not trying to replace answer evaluation, tracing, or experiment management. It is the diagnosis layer that sits between them and the engineering decisions that follow.

## Roadmap

- `v1.0`: deterministic diagnosis core, CLI, SDK, taxonomy, and baseline analyzers
- `v1.1`: stronger connectors, richer plugin configuration, and improved operational integration
- `v1.5`: deeper grounding, contradiction detection, and entropy-informed uncertainty signals
- `v2`: broader attribution coverage, stronger parser and chunking diagnosis, and production-scale diagnosis workflows

## Contributing

See [CONTRIBUTING.md](/Users/nitin/Desktop/RagGov/CONTRIBUTING.md).

## License

MIT
