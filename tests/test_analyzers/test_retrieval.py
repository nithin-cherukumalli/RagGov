"""Tests for deterministic retrieval analyzers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str, source_doc_id: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id or chunk_id,
        score=None,
    )


def run_with_chunks(
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is the refund policy?",
    cited_doc_ids: list[str] | None = None,
    corpus_entries: list[CorpusEntry] | None = None,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer="Answer.",
        cited_doc_ids=cited_doc_ids or [],
        corpus_entries=corpus_entries or [],
    )


def test_stale_retrieval_skips_without_chunks_or_corpus_metadata() -> None:
    analyzer = StaleRetrievalAnalyzer()

    empty_result = analyzer.analyze(run_with_chunks([]))
    no_metadata_result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "text")]))

    assert empty_result.status == "skip"
    assert empty_result.evidence == ["no retrieved chunks available"]
    assert no_metadata_result.status == "skip"
    assert no_metadata_result.evidence == ["no corpus metadata available"]


def test_stale_retrieval_fails_for_documents_past_age_threshold() -> None:
    old_doc = CorpusEntry(
        doc_id="doc-old",
        text="old text",
        timestamp=datetime.now(UTC) - timedelta(days=45),
    )
    fresh_doc = CorpusEntry(
        doc_id="doc-fresh",
        text="fresh text",
        timestamp=datetime.now(UTC) - timedelta(days=3),
    )
    run = run_with_chunks(
        [
            chunk("chunk-old", "old chunk", "doc-old"),
            chunk("chunk-fresh", "fresh chunk", "doc-fresh"),
        ],
        corpus_entries=[old_doc, fresh_doc],
    )

    result = StaleRetrievalAnalyzer({"max_age_days": 30}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.STALE_RETRIEVAL
    assert result.stage == FailureStage.RETRIEVAL
    assert result.evidence == ["doc-old is 45 days old"]
    assert result.remediation == (
        "Re-index documents older than 30 days or add freshness filtering to retrieval."
    )


def test_citation_mismatch_skips_without_chunks_or_citations() -> None:
    analyzer = CitationMismatchAnalyzer()

    empty_result = analyzer.analyze(run_with_chunks([]))
    no_citation_result = analyzer.analyze(run_with_chunks([chunk("chunk-1", "text")]))

    assert empty_result.status == "skip"
    assert empty_result.evidence == ["no retrieved chunks available"]
    assert no_citation_result.status == "skip"
    assert no_citation_result.evidence == ["no cited_doc_ids provided"]


def test_citation_mismatch_fails_for_citations_outside_retrieved_context() -> None:
    run = run_with_chunks(
        [chunk("chunk-1", "refund policy", "doc-1")],
        cited_doc_ids=["doc-1", "doc-phantom"],
    )

    result = CitationMismatchAnalyzer().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.CITATION_MISMATCH
    assert result.stage == FailureStage.RETRIEVAL
    assert result.evidence == ["phantom citation: doc-phantom"]
    assert result.remediation == (
        "Audit citation logic. Model cited documents outside the retrieved context window."
    )


def test_inconsistent_chunks_warns_on_nearby_negation_signals() -> None:
    run = run_with_chunks(
        [
            chunk("chunk-1", "The refund policy applies to hardware returns."),
            chunk("chunk-2", "Refund policy does not apply to hardware returns."),
        ]
    )

    result = InconsistentChunksAnalyzer().analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.INCONSISTENT_CHUNKS
    assert result.stage == FailureStage.RETRIEVAL
    assert result.evidence == ["chunk-1 <-> chunk-2"]
    assert result.remediation == (
        "Review retrieved chunks for contradictory information. "
        "Consider deduplication or reranking."
    )


def test_inconsistent_chunks_skips_empty_chunk_list() -> None:
    result = InconsistentChunksAnalyzer().analyze(run_with_chunks([]))

    assert result.status == "skip"
    assert result.evidence == ["no retrieved chunks available"]


def test_scope_violation_warns_for_some_off_topic_chunks() -> None:
    run = run_with_chunks(
        [
            chunk("chunk-1", "Refund policy covers returns and credits."),
            chunk("chunk-2", "Server firmware patch notes mention kernel drivers."),
        ],
        query="refund policy returns",
    )

    result = ScopeViolationAnalyzer({"min_overlap_ratio": 0.5}).analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.SCOPE_VIOLATION
    assert result.stage == FailureStage.RETRIEVAL
    assert result.evidence == ["chunk-2 overlap=0.00"]
    assert result.remediation == (
        "Retrieval returned off-topic documents. Review embedding model, "
        "query preprocessing, or index quality."
    )


def test_scope_violation_fails_when_all_chunks_are_off_topic() -> None:
    run = run_with_chunks(
        [chunk("chunk-1", "Server firmware patch notes mention kernel drivers.")],
        query="refund policy returns",
    )

    result = ScopeViolationAnalyzer({"min_overlap_ratio": 0.5}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.SCOPE_VIOLATION
    assert result.stage == FailureStage.RETRIEVAL
    assert result.evidence == ["chunk-1 overlap=0.00"]


def test_scope_violation_skips_empty_chunks_or_empty_query_terms() -> None:
    analyzer = ScopeViolationAnalyzer()

    empty_chunks = analyzer.analyze(run_with_chunks([]))
    empty_terms = analyzer.analyze(run_with_chunks([chunk("chunk-1", "text")], query="and or the"))

    assert empty_chunks.status == "skip"
    assert empty_chunks.evidence == ["no retrieved chunks available"]
    assert empty_terms.status == "skip"
    assert empty_terms.evidence == ["no meaningful query terms available"]
