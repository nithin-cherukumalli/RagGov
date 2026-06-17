"""Minimal GovRAG integration for a LlamaIndex RAG pipeline.

Run without LlamaIndex installed:

    python examples/llamaindex_integration.py

To wire a real LlamaIndex retriever/query engine, install it separately:

    pip install llama-index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raggov import RAGRun, RetrievedChunk
from raggov.engine import DiagnosisEngine
from raggov.io.serialize import diagnosis_why_block

try:
    from llama_index.core.schema import NodeWithScore
except ImportError:  # pragma: no cover - exercised by the runnable stub path
    NodeWithScore = None


@dataclass
class StubNode:
    text: str
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text


@dataclass
class StubNodeWithScore:
    node: StubNode
    score: float


class StubRetriever:
    """Small stand-in for a LlamaIndex retriever's retrieve() method."""

    def retrieve(self, query: str) -> list[Any]:
        node = StubNode(
            text="Refunds are available within fourteen days of purchase.",
            node_id="li-node-0",
            metadata={"source_doc_id": "policy-refunds"},
        )
        return [StubNodeWithScore(node=node, score=0.91)]


def answer_with_query_engine(query: str, nodes: list[Any]) -> str:
    """Replace this with your LlamaIndex query_engine.query(...) call."""
    return "Refunds are available within fourteen days of purchase."


def llamaindex_nodes_to_chunks(nodes: list[Any]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for index, node_with_score in enumerate(nodes):
        node = node_with_score.node
        metadata = dict(getattr(node, "metadata", {}) or {})
        text = node.get_content() if hasattr(node, "get_content") else str(node.text)
        chunks.append(
            RetrievedChunk(
                chunk_id=str(metadata.get("chunk_id", getattr(node, "node_id", f"li-{index}"))),
                text=text,
                source_doc_id=str(metadata.get("source_doc_id", metadata.get("file_name", f"doc-{index}"))),
                score=getattr(node_with_score, "score", None),
                metadata=metadata,
            )
        )
    return chunks


def main() -> None:
    query = "What is the refund window?"
    retriever = StubRetriever()
    nodes = retriever.retrieve(query)
    answer = answer_with_query_engine(query, nodes)

    run = RAGRun(
        query=query,
        retrieved_chunks=llamaindex_nodes_to_chunks(nodes),
        final_answer=answer,
        cited_doc_ids=["policy-refunds"],
    )
    diagnosis = DiagnosisEngine(config={"mode": "native"}).diagnose(run)
    why_block = diagnosis_why_block(diagnosis)

    print("primary_failure:", diagnosis.primary_failure.value)
    print("why:", why_block["verdict_summary"])


if __name__ == "__main__":
    main()
