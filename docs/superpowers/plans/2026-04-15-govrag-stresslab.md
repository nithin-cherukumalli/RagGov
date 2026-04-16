# GovRAG Stresslab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `stresslab/` package inside this repository that runs end-to-end RAG ingestion, chunking, embedding, retrieval, and answering against the government PDF corpus, then diagnoses each run with `raggov`.

**Architecture:** `stresslab/` is an internal standalone app with explicit stage boundaries. It produces structured parse artifacts, chunk artifacts, embedding/index artifacts, retrieval traces, answer traces, and a normalized `RAGRun` for `raggov`. Local file-backed indexing keeps the harness reproducible, while the LLM and embedding services remain pluggable over HTTP through LAN/public profiles.

**Tech Stack:** Python 3.10+, Pydantic v2, Typer, Rich, local JSON/JSONL artifacts, NumPy for vector math, pdfplumber or pdfminer.six for PDF extraction, pytest, ruff, mypy.

---

## File Structure

### New package

- Create: `stresslab/__init__.py`
- Create: `stresslab/config/__init__.py`
- Create: `stresslab/config/models.py`
- Create: `stresslab/config/load.py`
- Create: `stresslab/ingest/__init__.py`
- Create: `stresslab/ingest/models.py`
- Create: `stresslab/ingest/pdf_extract.py`
- Create: `stresslab/ingest/parse_go_order.py`
- Create: `stresslab/chunking/__init__.py`
- Create: `stresslab/chunking/base.py`
- Create: `stresslab/chunking/hierarchical.py`
- Create: `stresslab/chunking/fixed.py`
- Create: `stresslab/chunking/semantic.py`
- Create: `stresslab/embeddings/__init__.py`
- Create: `stresslab/embeddings/client.py`
- Create: `stresslab/index/__init__.py`
- Create: `stresslab/index/store.py`
- Create: `stresslab/retrieval/__init__.py`
- Create: `stresslab/retrieval/retrieve.py`
- Create: `stresslab/answering/__init__.py`
- Create: `stresslab/answering/client.py`
- Create: `stresslab/answering/prompting.py`
- Create: `stresslab/cases/__init__.py`
- Create: `stresslab/cases/models.py`
- Create: `stresslab/cases/load.py`
- Create: `stresslab/runners/__init__.py`
- Create: `stresslab/runners/ingest.py`
- Create: `stresslab/runners/build_index.py`
- Create: `stresslab/runners/run_case.py`
- Create: `stresslab/runners/run_suite.py`
- Create: `stresslab/reports/__init__.py`
- Create: `stresslab/reports/write.py`

### New configuration and artifacts

- Create: `stresslab/config/profiles/lan.json`
- Create: `stresslab/config/profiles/public.json`
- Create: `stresslab/cases/fixtures/parse_hierarchy_loss_ms20.json`
- Create: `stresslab/cases/fixtures/parse_table_corruption_ms39.json`
- Create: `stresslab/cases/fixtures/metadata_misread_ms1.json`
- Create: `stresslab/cases/fixtures/chunk_boundary_split_ms11.json`
- Create: `stresslab/cases/fixtures/undersegmentation_ms20.json`
- Create: `stresslab/cases/fixtures/oversegmentation_ms15.json`
- Create: `stresslab/cases/fixtures/embedding_semantic_drift_duplicates.json`
- Create: `stresslab/cases/fixtures/embedding_structured_relationship_ms9.json`
- Create: `stresslab/cases/fixtures/retrieval_missing_critical_context_ms20.json`
- Create: `stresslab/cases/fixtures/retrieval_ranking_instability_duplicate_cluster.json`
- Create: `stresslab/cases/fixtures/cross_section_reasoning_ms20.json`
- Create: `stresslab/cases/fixtures/abstention_required_private_fact.json`
- Create: `stresslab/artifacts/.gitkeep`
- Create: `stresslab/README.md`

### Project integration

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `src/raggov/__init__.py` only if a clean export hook is genuinely needed for stresslab integration

### Tests

- Create: `tests/stresslab/test_config.py`
- Create: `tests/stresslab/test_ingest.py`
- Create: `tests/stresslab/test_chunking.py`
- Create: `tests/stresslab/test_embeddings.py`
- Create: `tests/stresslab/test_index.py`
- Create: `tests/stresslab/test_retrieval.py`
- Create: `tests/stresslab/test_answering.py`
- Create: `tests/stresslab/test_cases.py`
- Create: `tests/stresslab/test_runners.py`
- Create: `tests/stresslab/test_integration.py`

## Task 1: Add Packaging and Runtime Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `stresslab/__init__.py`
- Test: `tests/stresslab/test_config.py`

- [ ] **Step 1: Write the failing packaging/config test**

```python
from stresslab import __all__


def test_stresslab_package_exports_version_marker() -> None:
    assert "__version__" in __all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/stresslab/test_config.py::test_stresslab_package_exports_version_marker -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stresslab'`

- [ ] **Step 3: Add minimal package skeleton and dependencies**

Update `pyproject.toml`:

```toml
[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]

[project.optional-dependencies]
stresslab = ["httpx>=0.27", "numpy>=1.26", "pdfplumber>=0.11"]
```

Create `stresslab/__init__.py`:

```python
"""GovRAG stress testing harness."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

- [ ] **Step 4: Run the targeted test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_config.py::test_stresslab_package_exports_version_marker -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml stresslab/__init__.py tests/stresslab/test_config.py
git commit -m "feat: scaffold stresslab package"
```

## Task 2: Add Config Profiles and Runtime Models

**Files:**
- Create: `stresslab/config/models.py`
- Create: `stresslab/config/load.py`
- Create: `stresslab/config/profiles/lan.json`
- Create: `stresslab/config/profiles/public.json`
- Test: `tests/stresslab/test_config.py`

- [ ] **Step 1: Write failing tests for config loading**

```python
from pathlib import Path

from stresslab.config.load import load_profile


def test_load_lan_profile() -> None:
    profile = load_profile("lan")
    assert profile.llm_base_url == "http://192.168.0.207:8000"
    assert str(profile.embedding_url).endswith("/v1/embeddings")


def test_load_profile_rejects_unknown_name() -> None:
    try:
        load_profile("missing")
    except ValueError as exc:
        assert "Unknown profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_config.py -v`
Expected: FAIL because `stresslab.config` does not exist

- [ ] **Step 3: Implement config models and loader**

Create `stresslab/config/models.py`:

```python
from pydantic import BaseModel, ConfigDict, HttpUrl


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    llm_base_url: HttpUrl
    embedding_url: HttpUrl
    answer_model: str
    embedding_model: str
    top_k: int = 5
    artifact_dir: str = "stresslab/artifacts"
```

Create `stresslab/config/load.py` with `load_profile(name: str) -> RuntimeProfile`.

- [ ] **Step 4: Add profile JSON files**

`stresslab/config/profiles/lan.json`

```json
{
  "name": "lan",
  "llm_base_url": "http://192.168.0.207:8000",
  "embedding_url": "http://192.168.0.207:8001/v1/embeddings",
  "answer_model": "TODO",
  "embedding_model": "TODO",
  "top_k": 5,
  "artifact_dir": "stresslab/artifacts"
}
```

`stresslab/config/profiles/public.json`

```json
{
  "name": "public",
  "llm_base_url": "http://103.44.12.98:8000",
  "embedding_url": "http://103.44.12.98:8001/v1/embeddings",
  "answer_model": "TODO",
  "embedding_model": "TODO",
  "top_k": 5,
  "artifact_dir": "stresslab/artifacts"
}
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add stresslab/config tests/stresslab/test_config.py
git commit -m "feat: add stresslab runtime profiles"
```

## Task 3: Define Structured Ingestion Models

**Files:**
- Create: `stresslab/ingest/models.py`
- Test: `tests/stresslab/test_ingest.py`

- [ ] **Step 1: Write failing model tests**

```python
from stresslab.ingest.models import ParsedDocument, ParsedNode, ParsedTable


def test_parsed_document_requires_lineage_fields() -> None:
    table = ParsedTable(table_id="t1", page=1, headers=["a"], rows=[["b"]])
    node = ParsedNode(node_id="n1", label="1", text="body", page_start=1, page_end=1)
    doc = ParsedDocument(
        doc_id="ms20",
        source_path="tests/Data/2011SE_MS20.PDF",
        title="Rules",
        abstract="abstract",
        department="School Education",
        go_number="20",
        issued_date="2011-03-03",
        nodes=[node],
        tables=[table],
    )
    assert doc.doc_id == "ms20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py::test_parsed_document_requires_lineage_fields -v`
Expected: FAIL because models do not exist

- [ ] **Step 3: Implement ingestion models**

Create Pydantic models for:

- `ParsedMetadata`
- `ParsedNode`
- `ParsedTable`
- `ParsedDocument`

Each model must forbid extra fields and preserve the hierarchy-oriented fields described in the spec.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py::test_parsed_document_requires_lineage_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/ingest/models.py tests/stresslab/test_ingest.py
git commit -m "feat: add parsed document models"
```

## Task 4: Build a Minimal PDF Extraction Layer

**Files:**
- Create: `stresslab/ingest/pdf_extract.py`
- Test: `tests/stresslab/test_ingest.py`

- [ ] **Step 1: Write failing extraction tests using the existing corpus**

```python
from pathlib import Path

from stresslab.ingest.pdf_extract import extract_pdf_text


def test_extract_pdf_text_reads_ms1() -> None:
    result = extract_pdf_text(Path("tests/Data/2011SE_MS1.PDF"))
    assert result.page_count == 1
    assert "Smt.T.Jyothi" in result.pages[0].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py::test_extract_pdf_text_reads_ms1 -v`
Expected: FAIL because extractor does not exist

- [ ] **Step 3: Implement extraction wrapper**

Implement a thin wrapper around `pdfplumber` returning:

- page count
- per-page text
- raw extracted lines if available

Keep this layer dumb. Parsing belongs in the next task.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py::test_extract_pdf_text_reads_ms1 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/ingest/pdf_extract.py tests/stresslab/test_ingest.py
git commit -m "feat: add pdf extraction layer"
```

## Task 5: Parse Government Order Structure

**Files:**
- Create: `stresslab/ingest/parse_go_order.py`
- Test: `tests/stresslab/test_ingest.py`

- [ ] **Step 1: Write failing parser tests against representative documents**

```python
from pathlib import Path

from stresslab.ingest.parse_go_order import parse_go_order


def test_parse_ms20_preserves_rule_hierarchy() -> None:
    parsed = parse_go_order(Path("tests/Data/2011SE_MS20.PDF"))
    labels = [node.label for node in parsed.nodes]
    assert "5" in labels
    assert any(node.label == "(4)" for node in parsed.nodes)


def test_parse_ms39_extracts_statement_table() -> None:
    parsed = parse_go_order(Path("tests/Data/2011SE_MS39.PDF"))
    assert any("Statement" in table.title for table in parsed.tables)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py -k "parse_ms20 or parse_ms39" -v`
Expected: FAIL because parser does not exist

- [ ] **Step 3: Implement minimal parser**

Implement:

- metadata extraction for title, abstract, department, G.O. number, issue date
- numbered clause detection
- sub-rule detection for patterns like `(1)`, `(a)`, `(i)`
- statement/annexure detection
- table block detection using text-line heuristics

Do not attempt a perfect parser here. Build a conservative parser that preserves line/section lineage.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_ingest.py -v`
Expected: PASS on the targeted parser tests

- [ ] **Step 5: Commit**

```bash
git add stresslab/ingest/parse_go_order.py tests/stresslab/test_ingest.py
git commit -m "feat: parse structured government orders"
```

## Task 6: Add Chunking Base Contracts and Fixed Chunker

**Files:**
- Create: `stresslab/chunking/base.py`
- Create: `stresslab/chunking/fixed.py`
- Test: `tests/stresslab/test_chunking.py`

- [ ] **Step 1: Write failing chunking tests**

```python
from stresslab.chunking.fixed import FixedChunker
from stresslab.ingest.models import ParsedDocument, ParsedNode


def test_fixed_chunker_emits_lineage() -> None:
    doc = ParsedDocument(
        doc_id="doc",
        source_path="doc.pdf",
        title="title",
        abstract="abstract",
        department="dept",
        go_number="1",
        issued_date="2011-01-01",
        nodes=[ParsedNode(node_id="n1", label="1", text="a b c d e f g", page_start=1, page_end=1)],
        tables=[],
    )
    chunks = FixedChunker(max_words=3, overlap_words=1).chunk(doc)
    assert chunks
    assert chunks[0].section_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_chunking.py::test_fixed_chunker_emits_lineage -v`
Expected: FAIL because chunker does not exist

- [ ] **Step 3: Implement chunk models and fixed chunker**

Define:

- `ChunkRecord`
- `BaseChunker`
- `FixedChunker.chunk(parsed_doc) -> list[ChunkRecord]`

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_chunking.py::test_fixed_chunker_emits_lineage -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/chunking tests/stresslab/test_chunking.py
git commit -m "feat: add chunking contracts and fixed chunker"
```

## Task 7: Add Hierarchical Chunker for Formal Government Orders

**Files:**
- Create: `stresslab/chunking/hierarchical.py`
- Test: `tests/stresslab/test_chunking.py`

- [ ] **Step 1: Write failing hierarchical chunking tests**

```python
from pathlib import Path

from stresslab.chunking.hierarchical import HierarchicalChunker
from stresslab.ingest.parse_go_order import parse_go_order


def test_hierarchical_chunker_keeps_rule_5_subrule_4_together() -> None:
    parsed = parse_go_order(Path("tests/Data/2011SE_MS20.PDF"))
    chunks = HierarchicalChunker().chunk(parsed)
    assert any("no school exists" in chunk.text.lower() for chunk in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_chunking.py::test_hierarchical_chunker_keeps_rule_5_subrule_4_together -v`
Expected: FAIL because hierarchical chunker does not exist

- [ ] **Step 3: Implement hierarchical chunker**

Rules:

- chunk at numbered clause boundaries
- keep sub-rules attached to their parent clause
- preserve statement/table blocks as standalone chunks
- include section path in every chunk

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_chunking.py -v`
Expected: PASS for hierarchical and fixed chunker tests

- [ ] **Step 5: Commit**

```bash
git add stresslab/chunking/hierarchical.py tests/stresslab/test_chunking.py
git commit -m "feat: add hierarchical chunker"
```

## Task 8: Add Embedding Client and Local Cache

**Files:**
- Create: `stresslab/embeddings/client.py`
- Test: `tests/stresslab/test_embeddings.py`

- [ ] **Step 1: Write failing client tests with mocked HTTP responses**

```python
from stresslab.embeddings.client import EmbeddingClient


def test_embedding_client_parses_openai_style_response(httpx_mock) -> None:
    httpx_mock.add_response(
        json={"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}
    )
    client = EmbeddingClient("http://localhost:8001/v1/embeddings", model="test")
    vectors = client.embed_texts(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_embeddings.py::test_embedding_client_parses_openai_style_response -v`
Expected: FAIL because client does not exist

- [ ] **Step 3: Implement client**

Implement:

- `embed_texts(texts: list[str]) -> list[list[float]]`
- local cache keyed by `(model, text hash)`
- timeout and clear error messages

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/embeddings tests/stresslab/test_embeddings.py
git commit -m "feat: add embedding client with cache"
```

## Task 9: Add Local Vector Index

**Files:**
- Create: `stresslab/index/store.py`
- Test: `tests/stresslab/test_index.py`

- [ ] **Step 1: Write failing index tests**

```python
from stresslab.index.store import LocalVectorIndex


def test_local_vector_index_returns_best_match() -> None:
    index = LocalVectorIndex()
    index.add("c1", [1.0, 0.0], {"text": "alpha"})
    index.add("c2", [0.0, 1.0], {"text": "beta"})
    results = index.search([1.0, 0.0], top_k=1)
    assert results[0].chunk_id == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_index.py::test_local_vector_index_returns_best_match -v`
Expected: FAIL because index does not exist

- [ ] **Step 3: Implement local index**

Use NumPy cosine similarity for:

- `add(chunk_id, vector, payload)`
- `search(query_vector, top_k)`
- `save(path)` / `load(path)`

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/index tests/stresslab/test_index.py
git commit -m "feat: add local vector index"
```

## Task 10: Add Retrieval Orchestration and Trace Model

**Files:**
- Create: `stresslab/retrieval/retrieve.py`
- Test: `tests/stresslab/test_retrieval.py`

- [ ] **Step 1: Write failing retrieval tests**

```python
from stresslab.retrieval.retrieve import RetrievalService


def test_retrieval_service_returns_ranked_chunks() -> None:
    service = RetrievalService(...)
    results = service.retrieve("query", top_k=2)
    assert len(results.chunks) == 2
    assert results.trace.query
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_retrieval.py -v`
Expected: FAIL because retrieval service does not exist

- [ ] **Step 3: Implement retrieval layer**

Implement:

- query embedding call
- index search
- trace object with scores and chunk lineage
- adapter from chunk records to `RetrievedChunk`

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/retrieval tests/stresslab/test_retrieval.py
git commit -m "feat: add retrieval service and trace"
```

## Task 11: Add Answering Client and Prompt Builder

**Files:**
- Create: `stresslab/answering/client.py`
- Create: `stresslab/answering/prompting.py`
- Test: `tests/stresslab/test_answering.py`

- [ ] **Step 1: Write failing answering tests**

```python
from stresslab.answering.prompting import build_prompt


def test_build_prompt_mentions_citation_requirement() -> None:
    prompt = build_prompt("What is Rule 5?", ["chunk text"])
    assert "cite" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_answering.py -v`
Expected: FAIL because answering package does not exist

- [ ] **Step 3: Implement prompt builder and LLM client**

Implement:

- deterministic prompt template for answer + cited doc ids
- OpenAI-compatible chat/completions HTTP wrapper for vLLM
- structured response adapter capturing answer text and citations when present

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_answering.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/answering tests/stresslab/test_answering.py
git commit -m "feat: add answering client and prompt builder"
```

## Task 12: Define Case Models and First Curated Scenario Files

**Files:**
- Create: `stresslab/cases/models.py`
- Create: `stresslab/cases/load.py`
- Create: `stresslab/cases/fixtures/*.json`
- Test: `tests/stresslab/test_cases.py`

- [ ] **Step 1: Write failing case validation tests**

```python
from stresslab.cases.load import load_case


def test_load_curated_case() -> None:
    case = load_case("parse_hierarchy_loss_ms20")
    assert case.expected_primary_failure == "PARSING"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_cases.py -v`
Expected: FAIL because case loader does not exist

- [ ] **Step 3: Implement case models and author scenario files**

Case files must capture:

- document set
- query
- gold answer
- supporting locations
- failure injection
- expected attribution

Start with the 12 scenarios from the approved spec.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_cases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/cases tests/stresslab/test_cases.py
git commit -m "feat: add curated GovRAG stress cases"
```

## Task 13: Build Ingest and Index Runners

**Files:**
- Create: `stresslab/runners/ingest.py`
- Create: `stresslab/runners/build_index.py`
- Create: `stresslab/reports/write.py`
- Test: `tests/stresslab/test_runners.py`

- [ ] **Step 1: Write failing runner tests**

```python
from pathlib import Path

from stresslab.runners.ingest import run_ingest


def test_run_ingest_writes_parsed_artifact(tmp_path: Path) -> None:
    artifact = run_ingest(
        source=Path("tests/Data/2011SE_MS1.PDF"),
        output_dir=tmp_path,
    )
    assert artifact.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_runners.py -k ingest -v`
Expected: FAIL because runner does not exist

- [ ] **Step 3: Implement ingest and index runners**

Implement:

- parse one or more PDFs into artifact files
- chunk parsed docs with a named strategy
- embed chunk sets
- build and persist local index
- record artifact manifest JSON

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_runners.py -v`
Expected: PASS for ingest/index tests

- [ ] **Step 5: Commit**

```bash
git add stresslab/runners stresslab/reports tests/stresslab/test_runners.py
git commit -m "feat: add ingest and index runners"
```

## Task 14: Build Single-Case Execution and `RAGRun` Integration

**Files:**
- Create: `stresslab/runners/run_case.py`
- Test: `tests/stresslab/test_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
from stresslab.runners.run_case import run_case


def test_run_case_returns_ragrun_and_diagnosis() -> None:
    result = run_case("parse_hierarchy_loss_ms20", profile="lan", dry_run=True)
    assert result.run.run_id
    assert result.diagnosis.run_id == result.run.run_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_integration.py::test_run_case_returns_ragrun_and_diagnosis -v`
Expected: FAIL because runner does not exist

- [ ] **Step 3: Implement single-case runner**

Implement:

- load case
- load profile
- run parse/chunk/index/retrieve/answer pipeline
- assemble `RAGRun`
- call `raggov.diagnose`
- persist run + diagnosis artifacts

Support `dry_run=True` with mocked answering for deterministic tests.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_integration.py::test_run_case_returns_ragrun_and_diagnosis -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/runners/run_case.py tests/stresslab/test_integration.py
git commit -m "feat: integrate stresslab runs with raggov diagnosis"
```

## Task 15: Build Batch Suite Execution and Summary Reporting

**Files:**
- Create: `stresslab/runners/run_suite.py`
- Modify: `stresslab/reports/write.py`
- Create: `stresslab/README.md`
- Modify: `README.md`
- Test: `tests/stresslab/test_integration.py`

- [ ] **Step 1: Write failing batch-run tests**

```python
from stresslab.runners.run_suite import run_suite


def test_run_suite_returns_case_summaries() -> None:
    report = run_suite(case_ids=["abstention_required_private_fact"], profile="lan", dry_run=True)
    assert report.total_cases == 1
    assert report.results[0].case_id == "abstention_required_private_fact"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_integration.py -k run_suite -v`
Expected: FAIL because suite runner does not exist

- [ ] **Step 3: Implement suite runner and summary reporting**

Implement:

- batch case execution
- per-case summaries
- aggregate failure matrix
- per-stage latency summary
- Markdown or JSON summary output

Document:

- how to run the suite
- how to select LAN/public profile
- where artifacts are written

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:. pytest tests/stresslab/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stresslab/runners/run_suite.py stresslab/reports/write.py stresslab/README.md README.md tests/stresslab/test_integration.py
git commit -m "feat: add batch stress suite and reporting"
```

## Task 16: Add End-to-End Verification and Quality Gates

**Files:**
- Modify: `tests/stresslab/test_integration.py`
- Modify: `stresslab/README.md`

- [ ] **Step 1: Add smoke tests covering the real corpus**

```python
def test_duplicate_cluster_case_runs_in_dry_mode() -> None:
    ...


def test_ms20_case_preserves_expected_attribution_fields() -> None:
    ...
```

- [ ] **Step 2: Run all stresslab tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. pytest tests/stresslab -q`
Expected: PASS

- [ ] **Step 3: Run project-wide checks**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. pytest -q`
Expected: PASS

Run: `python -m ruff check src tests stresslab`
Expected: PASS

Run: `PYTHONPATH=src:. python -m mypy src tests stresslab`
Expected: PASS

Run: `python -m compileall -q src tests stresslab`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/stresslab stresslab/README.md
git commit -m "test: verify stresslab end-to-end slice"
```

## Notes for the Implementer

- Prefer deterministic tests with mocked HTTP clients. Only add live-service smoke checks behind explicit opt-in flags.
- Keep `stresslab` independent from `raggov` internals. Use public models and `raggov.diagnose`.
- Preserve artifact lineage aggressively. If a chunk cannot be traced back to page, section, and parse node, the design is losing the core requirement.
- Do not add a full vector database yet. Local file-backed indexing is enough for the first working slice.
- Do not attempt perfect table parsing in the first pass. Preserve enough line and row structure to support failure-driven evaluation.

## Review Constraint

This plan should normally go through a subagent review loop, but delegation is currently restricted in this session unless explicitly requested by the user. If review is required before execution, ask for permission to use a review subagent.
