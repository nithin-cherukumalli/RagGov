# GovRAG Stresslab Design

Date: 2026-04-15

## Goal

Build a full-fledged, repeatable RAG stress harness inside this repository that exercises real ingestion, chunking, embedding, retrieval, and answering stages against the government-document corpus in `tests/Data`, then feeds each run into `raggov` for diagnosis.

The harness must remain self-contained for now, but the package boundary must be clean enough that it can later move into its own repository without changing the `raggov` integration contract.

## Why This Shape

`raggov` is already the diagnosis layer. What is missing is a realistic upstream RAG system that can fail in controlled, attributable ways. Extending `tests/` would mix regression tests with service orchestration and data pipelines. Building a separate internal package gives us a real system boundary now and a clean extraction path later.

## Corpus Findings

The corpus in `tests/Data` is narrow, formal, and structurally repetitive:

- Several one-page government orders with nearly identical procedural wording and different entities or districts
- Medium-length orders with statements, annexures, and tabular allocations
- One long rules document with nested numbered sections and sub-rules

These patterns are exactly what a failure-driven RAG harness needs:

- Near-duplicate documents expose embedding collapse and retrieval ranking mistakes
- Tables expose parser and chunking weakness
- Deeply nested rules expose hierarchy loss and boundary errors

### Key Documents

- `2011SE_MS20.PDF`: long hierarchical rules document, anchor for hierarchy and cross-section reasoning
- `2011SE_MS9.PDF`: annexure-heavy document with district-wise numeric allocations
- `2011SE_MS15.PDF`: pay anomaly document with category/pay-scale table structure
- `2011SE_MS39.PDF`: staffing reorganization order with multiple structured statements and table blocks
- `2011SE_MS24.PDF`, `2011SE_MS29.PDF`, `2011SE_MS30.PDF`, `2011SE_MS35.PDF`: near-duplicate vigilance orders, anchor cluster for embedding and retrieval stress
- `2011SE_MS11.PDF`: ordered narrative plus embedded teacher list, useful for chunk-boundary tests
- `2011SE_MS1.PDF`: short order with metadata and rule-reference dependency

## Architecture

Create a new top-level internal package named `stresslab/`.

`stresslab/` behaves like an external app that happens to live in this repository. It imports `raggov` through a clean API boundary instead of reaching into internals. Every end-to-end run must emit a normalized `RAGRun`, then call `raggov.diagnose(run)`.

### Module Layout

- `stresslab/config/`
  - runtime profiles for LAN/public endpoints
  - model names, top-k, batch size, output locations
- `stresslab/ingest/`
  - PDF extraction
  - metadata normalization
  - hierarchy parsing
  - table extraction
- `stresslab/chunking/`
  - hierarchical chunker
  - semantic chunker
  - fixed-size chunker
- `stresslab/embeddings/`
  - client for `/v1/embeddings`
  - batching, retries, local cache
- `stresslab/index/`
  - local file-backed vector index
  - ranking helpers
- `stresslab/retrieval/`
  - query embedding
  - top-k retrieval
  - retrieval trace capture
- `stresslab/answering/`
  - LLM client for `:8000`
  - prompt assembly
  - answer trace and citation capture
- `stresslab/cases/`
  - scenario definitions
  - gold parse expectations
  - gold answers
  - expected failure labels
- `stresslab/runners/`
  - ingest runner
  - build-index runner
  - single-case runner
  - batch suite runner
- `stresslab/reports/`
  - raw run artifacts
  - diagnosis outputs
  - benchmark summaries
  - comparison reports

### Data Flow

`PDF -> structured document JSON -> chunk set -> embeddings -> local index -> retrieval -> answer -> RAGRun -> raggov Diagnosis`

## Design Principles

### 1. Stage Boundaries Must Be Explicit

GovRAG is only useful if it can say where failure originated. Each stage must emit its own artifact and lineage metadata, so later failure attribution can distinguish:

- parser lost structure
- chunker split evidence badly
- embedder failed to preserve semantics
- retriever ranked the wrong chunk
- answerer hallucinated beyond evidence

### 2. Local-First, Service-Pluggable

The harness should use the local embedding and LLM services over HTTP, but keep everything else local and file-backed. A separate vector database is not needed initially. Local indexing keeps the system reproducible and debuggable.

### 3. Corpus-Tailored, Not Benchmark-Themed

The cases should be derived from the actual government-order corpus instead of synthetic generic QA. The failure model must follow the structure of these documents: formal headers, read-lists, numbered clauses, annexures, tables, and repeated procedural language.

## Gold Artifacts

For each selected source document, generate a canonical set of artifacts:

### Parsed Document JSON

Must preserve:

- `doc_id`
- title and abstract
- department
- G.O. number
- issue date
- read-list / references
- ordered clauses
- section tree
- table blocks as structured rows/cells
- distribution list

For `MS20`, the parser must retain:

- rule number
- sub-rule nesting
- clause markers like `(1)`, `(a)`, `(i)`
- definitions versus operative provisions

For `MS9`, `MS15`, and `MS39`, the parser must retain:

- annexure/statement identity
- row boundaries
- cell groupings
- district/category to value mappings

### Chunk Artifacts

Each chunk record must preserve lineage:

- `chunk_id`
- `source_doc_id`
- `page_range`
- `section_path`
- `table_id` where applicable
- `chunk_strategy`
- `parent_parse_node_id`
- text content

### Query and Answer Gold Files

For each case:

- canonical query
- phrasing variants
- gold answer
- supporting sections / pages / table rows
- whether the system should abstain if context is missing
- expected primary failure under injected conditions

## Chunking Strategy Evaluation

Three chunkers will be evaluated on the same corpus and cases.

### Hierarchical Chunking

Split on:

- major headers
- numbered paragraphs
- rule/sub-rule boundaries
- annexures/statements
- table blocks

This should be the strongest baseline for `MS20`, `MS9`, and `MS39`.

### Semantic Chunking

Split on semantically coherent paragraph groups after normalization. This is useful for narrative documents, but risky on legal and tabular structure because semantic coherence does not preserve formal scope.

### Fixed-Size Chunking

Use token windows with overlap as a control baseline. This is expected to fail on boundary-sensitive cases and gives GovRAG a clear chance to attribute chunking defects.

## Failure-Driven Case Model

Each case definition should include:

- `case_id`
- `document_set`
- `query`
- `gold_answer`
- `gold_supporting_locations`
- `expected_structured_parse`
- `pipeline_variant`
- `failure_injection`
- `expected_primary_failure`
- `expected_secondary_failures`
- `expected_should_have_answered`
- `severity`
- `deterministic_or_query_dependent`

## Tailored Failure Classes

### Parsing Failures

- lost heading depth
- flattened rule/sub-rule hierarchy
- table-to-text corruption
- annexure identity loss
- metadata extraction error

### Chunking Failures

- legal condition split across chunks
- table row split from header
- oversegmentation of supporting context
- undersegmentation mixing unrelated provisions
- duplicate overlap causing evidence bloat

### Embedding Failures

- near-duplicate orders collapse together
- district or entity names underweighted relative to procedural boilerplate
- tabular relationships poorly represented
- cross-section dependencies poorly represented

### Retrieval Failures

- wrong near-duplicate ranked first
- missing critical clause from top-k
- ranking instability across phrasing variants
- retrieval of definitions instead of operative clauses

### Answering Failures

- unsupported synthesis
- wrong entity substitution
- citation mismatch
- failure to abstain

## First Curated Scenario Set

### 1. `parse_hierarchy_loss_ms20`

- Document: `MS20`
- Query: what accommodations are required when no neighborhood school exists within the specified limits
- Injection: flatten Rule 5 sub-rules and remove the explicit boundary around sub-rule `(4)`
- Expected origin: parsing
- Expected condition: loss of parent-child rule structure

### 2. `parse_table_corruption_ms39`

- Document: `MS39`
- Query: what is the proposed Project Officer pattern for Warangal and Khammam
- Injection: collapse Statement-A columns into plain text
- Expected origin: parsing
- Expected condition: row/column association loss

### 3. `metadata_misread_ms1`

- Document: `MS1`
- Query: on what date was the reinstatement order issued
- Injection: wrong date extraction or G.O. number extraction
- Expected origin: parsing

### 4. `chunk_boundary_split_ms11`

- Document: `MS11`
- Query: why were the four teachers repatriated and under which authority
- Injection: split para 2 and para 3 into separate minimal chunks without enough overlap
- Expected origin: chunking

### 5. `undersegmentation_ms20`

- Document: `MS20`
- Query: what does the rule say about children with disabilities accessing school
- Injection: large fixed chunks mixing definitions, access rules, and state duties
- Expected origin: chunking

### 6. `oversegmentation_ms15`

- Document: `MS15`
- Query: which categories were identified for rectification and which had financial implications
- Injection: row-per-line chunking across the table and explanatory paragraph
- Expected origin: chunking

### 7. `embedding_semantic_drift_duplicates`

- Documents: `MS24`, `MS29`, `MS30`, `MS35`
- Query: which retired teacher from Nalgonda District was sanctioned proceedings under Rule 9
- Stress: near-identical language with different entities and districts
- Expected origin if wrong: embedding or retrieval

### 8. `embedding_structured_relationship_ms9`

- Document: `MS9`
- Query: how many posts can be adjusted to SSA for Krishna district
- Stress: district-to-column relationship in annexure
- Expected origin if wrong: embedding or chunking

### 9. `retrieval_missing_critical_context_ms20`

- Document: `MS20`
- Query: what is the minimum period of special training and how long may it be extended
- Injection: retrieval constrained to definition-heavy chunks or low top-k
- Expected origin: retrieval

### 10. `retrieval_ranking_instability_duplicate_cluster`

- Documents: `MS24`, `MS29`, `MS30`, `MS35`
- Query variants target district and role combinations across nearly identical orders
- Expected origin if wrong: retrieval

### 11. `cross_section_reasoning_ms20`

- Document: `MS20`
- Query: how do the rules handle children with severe disability when neighborhood access is not feasible
- Requires combining Rule 5(7) with surrounding access provisions
- Expected origin if wrong: chunking, retrieval, or grounding depending on trace

### 12. `abstention_required_private_fact`

- Corpus: entire set
- Query: what budget was approved in 2012 for district-level implementation of Rule 5
- Gold answer: abstain
- Expected diagnosis: insufficient context
- Expected should-have-answered: false

## Failure Classification Matrix

| Scenario | Primary Origin | Condition | Severity | Deterministic |
| --- | --- | --- | --- | --- |
| parse_hierarchy_loss_ms20 | Parsing | lost rule hierarchy | High | Yes |
| parse_table_corruption_ms39 | Parsing | row/column loss | High | Yes |
| metadata_misread_ms1 | Parsing | wrong metadata | Medium | Yes |
| chunk_boundary_split_ms11 | Chunking | evidence split across chunks | High | Yes |
| oversegmentation_ms15 | Chunking | context fragmentation | Medium | Yes |
| undersegmentation_ms20 | Chunking | mixed-topic chunk dilution | Medium | No |
| embedding_semantic_drift_duplicates | Embedding | near-duplicate collapse | High | No |
| embedding_structured_relationship_ms9 | Embedding | table semantics not preserved | High | No |
| retrieval_missing_critical_context_ms20 | Retrieval | critical clause absent from top-k | High | No |
| retrieval_ranking_instability_duplicate_cluster | Retrieval | wrong duplicate ranked first | High | No |
| abstention_required_private_fact | Sufficiency | no supporting evidence | High | Yes |

## Evaluation Outputs

For each case, the suite must produce:

- raw parse artifact
- chunk set artifact
- embedding artifact
- retrieval trace
- answer output
- normalized `RAGRun`
- `Diagnosis`
- comparison record against gold expectations

## Observability Requirements

### Parse Stage

- parse time
- extracted metadata
- section tree depth
- table count
- parse warnings

### Chunk Stage

- chunk count
- token distribution
- section coverage
- table-preserved flag
- overlap statistics

### Embedding Stage

- embedding model name
- request latency
- batch size
- vector dimension
- cache hit/miss

### Retrieval Stage

- query text
- query embedding latency
- top-k chunk ids
- scores
- score gaps
- rank changes across phrasing variants

### Answer Stage

- prompt template id
- retrieved chunk ids
- generation latency
- answer text
- cited doc ids

### Diagnosis Stage

- primary failure
- secondary failures
- root cause stage
- should have answered
- security risk
- confidence
- evidence

## Endpoint Profiles

Support profile-based execution:

### LAN

- LLM: `http://192.168.0.207:8000`
- Embeddings: `http://192.168.0.207:8001/v1/embeddings`

### Public

- LLM: `http://103.44.12.98:8000`
- Embeddings: `http://103.44.12.98:8001/v1/embeddings`

The suite should use a single runtime switch to select profile, not hard-coded URLs scattered through the codebase.

## Reproducibility

Each suite run should record:

- git revision
- endpoint profile
- embedding model
- answer model
- chunking strategy
- top-k
- corpus selection
- timestamp

This metadata belongs in the run report so results remain comparable over time.

## Recommended First Implementation Slice

Phase 1 should build the minimum full loop:

1. `stresslab/` package skeleton
2. config loader with LAN/public profiles
3. PDF extraction and normalized structured document format
4. hierarchical chunker and fixed-size chunker
5. embedding client for `:8001`
6. local vector index
7. retrieval trace capture
8. answer client for `:8000`
9. single-case runner producing `RAGRun`
10. integration with `raggov.diagnose`
11. first 10 curated cases from `MS1`, `MS9`, `MS11`, `MS15`, `MS20`, `MS24`, `MS29`, `MS35`, `MS39`

Phase 2 should add:

- semantic chunking
- phrasing-variation benchmarks
- batch suite runner
- failure injection toggles
- richer comparison reports

## Recommendation

Proceed with `stresslab/` as an internal standalone app inside this repository, using local file-backed indexing and the provided HTTP services for embeddings and generation. This is the highest-leverage starting point because it preserves clean architecture, supports real end-to-end stress testing, and does not lock the project into infrastructure choices too early.
