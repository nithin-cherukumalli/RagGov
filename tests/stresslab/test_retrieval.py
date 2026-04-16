import pytest

from stresslab.index import SearchResult, VectorIndex
from stresslab.retrieval import RetrievalResult, RetrievalService, RetrievalTrace


class StubEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = list(vectors)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return list(self._vectors)


def test_retrieval_service_returns_ranked_chunks_and_trace() -> None:
    index = VectorIndex()
    index.add(
        "chunk-1",
        [1.0, 0.0],
        {
            "text": "Relevant rule text",
            "source_doc_id": "doc-1",
            "page_start": 2,
            "page_end": 3,
            "section_path": ["Rules", "Rule 5"],
            "parent_node_id": "node-1",
        },
    )
    index.add(
        "chunk-2",
        [0.5, 0.5],
        {
            "text": "Related but weaker match",
            "source_doc_id": "doc-2",
            "page_start": 4,
            "page_end": 4,
            "section_path": "Appendix",
            "parent_node_id": "node-2",
        },
    )
    service = RetrievalService(
        embedding_client=StubEmbeddingClient([[1.0, 0.0]]),
        index=index,
    )

    result = service.retrieve("What is Rule 5?", top_k=2)

    assert isinstance(result, RetrievalResult)
    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1", "chunk-2"]
    assert result.chunks[0].text == "Relevant rule text"
    assert result.chunks[0].source_doc_id == "doc-1"
    assert result.chunks[0].score == pytest.approx(1.0)
    assert result.chunks[0].metadata == {
        "page_start": 2,
        "page_end": 3,
        "section_path": ["Rules", "Rule 5"],
        "parent_node_id": "node-1",
    }
    assert isinstance(result.trace, RetrievalTrace)
    assert result.trace.query == "What is Rule 5?"
    assert result.trace.query_vector == [1.0, 0.0]
    assert result.trace.top_k == 2
    assert result.trace.results == [
        SearchResult(
            chunk_id="chunk-1",
            score=pytest.approx(1.0),
            payload={
                "text": "Relevant rule text",
                "source_doc_id": "doc-1",
                "page_start": 2,
                "page_end": 3,
                "section_path": ["Rules", "Rule 5"],
                "parent_node_id": "node-1",
            },
        ),
        SearchResult(
            chunk_id="chunk-2",
            score=pytest.approx(0.70710678, rel=1e-6),
            payload={
                "text": "Related but weaker match",
                "source_doc_id": "doc-2",
                "page_start": 4,
                "page_end": 4,
                "section_path": "Appendix",
                "parent_node_id": "node-2",
            },
        ),
    ]
    assert service._embedding_client.calls == [["What is Rule 5?"]]


def test_retrieval_service_uses_chunk_id_when_source_doc_id_missing() -> None:
    index = VectorIndex()
    index.add(
        "chunk-1",
        [1.0],
        {
            "text": "Fallback source id",
            "page_start": 1,
            "page_end": 1,
        },
    )
    service = RetrievalService(
        embedding_client=StubEmbeddingClient([[1.0]]),
        index=index,
    )

    result = service.retrieve("query", top_k=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "chunk-1"
    assert result.chunks[0].source_doc_id == "chunk-1"
    assert result.chunks[0].metadata == {"page_start": 1, "page_end": 1}
