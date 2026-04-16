"""Single-case stresslab runner."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from raggov import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun

from stresslab.answering import AnsweringClient
from stresslab.cases import load_case
from stresslab.chunking import ChunkRecord, HierarchicalChunker
from stresslab.config import RuntimeProfile, load_profile
from stresslab.embeddings import EmbeddingClient
from stresslab.index import VectorIndex
from stresslab.ingest import ParsedDocument, parse_go_order
from stresslab.retrieval import RetrievalService
from stresslab.mutations import (
    flatten_hierarchy,
    collapse_tables,
    duplicate_chunks,
    constrain_top_k,
    oversegment,
    undersegment,
)


_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_DATA_DIR = _ROOT_DIR / "tests" / "Data"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_DRY_RUN_DIMENSION = 32
_PRIVATE_QUERY_PATTERNS = (
    "address",
    "phone",
    "email",
    "private",
    "personal",
    "home",
    "ssn",
)


class _EmbeddingClientLike(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class RunCaseResult:
    case_id: str
    profile_name: str
    run: RAGRun
    diagnosis: Diagnosis


class _DryRunEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * _DRY_RUN_DIMENSION
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % _DRY_RUN_DIMENSION
            weight = 1.0 + (digest[1] / 255.0)
            vector[bucket] += weight
        if not any(vector):
            vector[0] = 1.0
        return vector

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


def run_case(case_id: str, profile: str, dry_run: bool = False) -> RunCaseResult:
    case = load_case(case_id)
    runtime_profile = load_profile(profile)

    parsed_documents = [_parse_source_document(name) for name in case.document_set]

    # Apply parser-level mutations if specified
    if case.failure_injection == "drop_parent_child_links":
        parsed_documents = [flatten_hierarchy(doc) for doc in parsed_documents]
    elif case.failure_injection == "flatten_statement_rows" or case.failure_injection == "collapse_statement_columns":
        parsed_documents = [collapse_tables(doc) for doc in parsed_documents]

    chunker = HierarchicalChunker()
    chunks = [
        chunk
        for parsed_document in parsed_documents
        for chunk in chunker.chunk(parsed_document)
    ]

    # Apply chunker-level mutations
    if case.failure_injection == "duplicate_near_identical_chunks":
        chunks = duplicate_chunks(chunks)
    elif case.failure_injection == "oversegmentation" or case.failure_injection == "oversegmentation_ms15":
        chunks = oversegment(chunks)
    elif case.failure_injection == "undersegmentation" or case.failure_injection == "undersegmentation_ms20":
        chunks = undersegment(chunks)

    embedding_client = _DryRunEmbeddingClient() if dry_run else _build_embedding_client(runtime_profile)
    try:
        index = _build_index(chunks, embedding_client)

        # Apply retrieval-level mutations
        retrieval_top_k = runtime_profile.top_k
        if case.failure_injection == "top_k_excludes_exception_chunk":
            # Reduce top-k to exclude comprehensive chunks
            retrieval_top_k = max(1, runtime_profile.top_k - 2)

        retrieval = RetrievalService(embedding_client=embedding_client, index=index).retrieve(
            case.query,
            top_k=retrieval_top_k,
        )

        if dry_run:
            final_answer = _synthesize_dry_run_answer(case.query, retrieval.chunks)
        else:
            with _build_answering_client(runtime_profile) as answering_client:
                final_answer = answering_client.answer(
                    case.query,
                    [f"[{chunk.source_doc_id}] {chunk.text}" for chunk in retrieval.chunks],
                )
    finally:
        close = getattr(embedding_client, "close", None)
        if callable(close):
            close()

    cited_doc_ids = (
        []
        if _looks_private_query(case.query)
        else _ordered_unique([chunk.source_doc_id for chunk in retrieval.chunks[:2]])
    )

    run = RAGRun(
        query=case.query,
        retrieved_chunks=retrieval.chunks,
        final_answer=final_answer,
        cited_doc_ids=cited_doc_ids,
        answer_confidence=_answer_confidence(retrieval.chunks, final_answer),
        trace={
            "case_id": case.case_id,
            "profile": runtime_profile.name,
            "dry_run": dry_run,
            "source_documents": [document.doc_id for document in parsed_documents],
            "retrieval": {
                "top_k": retrieval.trace.top_k,
                "query_vector": retrieval.trace.query_vector,
                "results": [
                    {
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                        "source_doc_id": (
                            result.payload.get("source_doc_id")
                            if isinstance(result.payload, dict)
                            else None
                        ),
                    }
                    for result in retrieval.trace.results
                ],
            },
        },
        metadata={
            "case_id": case.case_id,
            "profile": runtime_profile.name,
            "dry_run": dry_run,
            "pipeline_variant": case.pipeline_variant,
            "failure_injection": case.failure_injection,
            "expected_primary_failure": case.expected_primary_failure,
        },
    )
    diagnosis = diagnose(run)
    return RunCaseResult(
        case_id=case.case_id,
        profile_name=runtime_profile.name,
        run=run,
        diagnosis=diagnosis,
    )


def _build_embedding_client(runtime_profile: RuntimeProfile) -> EmbeddingClient:
    return EmbeddingClient(
        base_url=str(runtime_profile.embedding_url),
        model=runtime_profile.embedding_model,
    )


def _build_answering_client(runtime_profile: RuntimeProfile) -> AnsweringClient:
    base_url = str(runtime_profile.llm_base_url).rstrip("/") + _CHAT_COMPLETIONS_PATH
    return AnsweringClient(base_url=base_url, model=runtime_profile.answer_model)


def _parse_source_document(name: str) -> ParsedDocument:
    source_path = _resolve_source_path(name)
    return parse_go_order(source_path)


def _resolve_source_path(name: str) -> Path:
    source = Path(name)
    if source.is_file():
        return source

    candidate = _TEST_DATA_DIR / name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Unable to resolve source document: {name}")


def _build_index(chunks: list[ChunkRecord], embedding_client: _EmbeddingClientLike) -> VectorIndex:
    index = VectorIndex()
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}"
        )
    for chunk, vector in zip(chunks, embeddings, strict=True):
        index.add(chunk.chunk_id, vector, chunk.model_dump(mode="json"))
    return index


def _synthesize_dry_run_answer(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "No matching context found."
    if _looks_private_query(query):
        return "Cannot answer from public context."

    top_chunk = retrieved_chunks[0]
    snippet = " ".join(top_chunk.text.split())
    if len(snippet) > 280:
        snippet = snippet[:277].rstrip() + "..."
    return f"{snippet} [{top_chunk.source_doc_id}]"


def _looks_private_query(query: str) -> bool:
    lowered = query.lower()
    return any(pattern in lowered for pattern in _PRIVATE_QUERY_PATTERNS)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _answer_confidence(retrieved_chunks: list[RetrievedChunk], final_answer: str) -> float:
    normalized_answer = final_answer.strip().lower()
    if not retrieved_chunks or normalized_answer.startswith("cannot answer") or normalized_answer.startswith(
        "no matching context"
    ):
        return 0.25
    top_score = retrieved_chunks[0].score or 0.0
    return round(min(0.99, max(0.25, (top_score + 1.0) / 2.0)), 2)
