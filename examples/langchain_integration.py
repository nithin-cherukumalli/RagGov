"""Minimal GovRAG integration for a LangChain RAG pipeline.

Run without LangChain installed:

    python examples/langchain_integration.py

To wire a real LangChain retriever, install LangChain separately:

    pip install langchain
"""

from __future__ import annotations

from typing import Any

from raggov import RAGRun, RetrievedChunk
from raggov.engine import DiagnosisEngine
from raggov.io.serialize import diagnosis_why_block

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - exercised by the runnable stub path
    Document = None


class StubRetriever:
    """Small stand-in for a LangChain retriever's invoke() method."""

    def invoke(self, query: str) -> list[Any]:
        if Document is not None:
            return [
                Document(
                    page_content="Refunds are available within fourteen days of purchase.",
                    metadata={"source_doc_id": "policy-refunds", "score": 0.91},
                )
            ]
        return [
            {
                "page_content": "Refunds are available within fourteen days of purchase.",
                "metadata": {"source_doc_id": "policy-refunds", "score": 0.91},
            }
        ]


def answer_with_llm(query: str, docs: list[Any]) -> str:
    """Replace this with your LangChain chain.invoke(...) call."""
    return "Refunds are available within fourteen days of purchase."


def langchain_docs_to_chunks(docs: list[Any]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for index, doc in enumerate(docs):
        metadata = getattr(doc, "metadata", None) or doc.get("metadata", {})
        text = getattr(doc, "page_content", None) or doc["page_content"]
        chunks.append(
            RetrievedChunk(
                chunk_id=str(metadata.get("chunk_id", f"lc-{index}")),
                text=text,
                source_doc_id=str(metadata.get("source_doc_id", metadata.get("source", f"doc-{index}"))),
                score=float(metadata["score"]) if metadata.get("score") is not None else None,
                metadata=dict(metadata),
            )
        )
    return chunks


def main() -> None:
    query = "What is the refund window?"
    retriever = StubRetriever()
    docs = retriever.invoke(query)
    answer = answer_with_llm(query, docs)

    run = RAGRun(
        query=query,
        retrieved_chunks=langchain_docs_to_chunks(docs),
        final_answer=answer,
        cited_doc_ids=["policy-refunds"],
    )
    diagnosis = DiagnosisEngine(config={"mode": "native"}).diagnose(run)
    why_block = diagnosis_why_block(diagnosis)

    print("primary_failure:", diagnosis.primary_failure.value)
    print("why:", why_block["verdict_summary"])


if __name__ == "__main__":
    main()
