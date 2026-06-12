"""Tests for relative-recency STALE_RETRIEVAL detection.

A senior RAG engineer manually inspects chunk metadata for `effective_date`,
`valid_until`, `as_of`, `published_at`, `updated_at`, `valid_from`, or `date`,
and concludes a retrieval is stale when:

  - the retrieved set contains at least two chunks with parseable temporal
    metadata, and
  - the answer's content textually aligns with an older chunk while a
    strictly newer chunk was also retrieved.

This is pipeline-agnostic (any retriever populating standard chunk metadata)
and domain-agnostic (tax, security, policy, HR — anything time-bound).
"""

from __future__ import annotations

from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


def _chunk(chunk_id: str, text: str, doc_id: str, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=doc_id,
        score=0.9,
        metadata=metadata or {},
    )


def _run(chunks: list[RetrievedChunk], *, query: str, answer: str) -> RAGRun:
    return RAGRun(query=query, retrieved_chunks=chunks, final_answer=answer)


def test_relative_recency_fires_when_answer_uses_older_chunk_over_newer() -> None:
    """Calib-50 case 039 pattern: 2026 query, 2025 rate in answer, newer chunk retrieved."""
    chunks = [
        _chunk(
            "c1",
            "The 2025 mileage reimbursement rate is $0.67 per mile.",
            "mileage-2025",
            {"effective_date": "2025-01-01", "valid_until": "2025-12-31"},
        ),
        _chunk(
            "c2",
            "The 2026 mileage reimbursement rate is $0.71 per mile.",
            "mileage-2026",
            {"effective_date": "2026-01-01"},
        ),
    ]
    run = _run(
        chunks,
        query="What is the 2026 mileage reimbursement rate?",
        answer="The 2026 mileage reimbursement rate is $0.67 per mile.",
    )

    result = StaleRetrievalAnalyzer().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.STALE_RETRIEVAL
    assert result.stage == FailureStage.RETRIEVAL
    assert any("c1" in e for e in result.evidence), result.evidence
    assert any("c2" in e for e in result.evidence), result.evidence


def test_relative_recency_passes_when_answer_uses_newest_chunk() -> None:
    """If the answer aligns with the newest chunk, the freshness is fine."""
    chunks = [
        _chunk("c1", "Rate was $0.67 in 2025.", "d1", {"effective_date": "2025-01-01"}),
        _chunk("c2", "Rate is $0.71 in 2026.", "d2", {"effective_date": "2026-01-01"}),
    ]
    run = _run(
        chunks,
        query="What is the 2026 rate?",
        answer="The 2026 rate is $0.71.",
    )

    result = StaleRetrievalAnalyzer().analyze(run)

    # Must not fire STALE_RETRIEVAL when the answer already used the newer chunk.
    assert result.failure_type != FailureType.STALE_RETRIEVAL


def test_relative_recency_passes_when_dates_are_identical() -> None:
    """Identical dates mean no freshness conflict (Calib-50 case 032 pattern)."""
    chunks = [
        _chunk("c1", "Audit logs retained 400 days.", "d1", {"effective_date": "2026-01-01"}),
        _chunk("c2", "Access logs retained 90 days.", "d2", {"effective_date": "2026-01-01"}),
    ]
    run = _run(
        chunks,
        query="What audit log retention period applies?",
        answer="Audit logs are retained for 400 days.",
    )

    result = StaleRetrievalAnalyzer().analyze(run)

    # Must not fire STALE_RETRIEVAL when dates are equal — staleness is
    # a relative-recency claim and equal dates carry no relative signal.
    assert result.failure_type != FailureType.STALE_RETRIEVAL


def test_relative_recency_accepts_published_at_and_valid_until_keys() -> None:
    """Recognised keys: effective_date, valid_until, valid_from, as_of, published_at, updated_at, date."""
    chunks = [
        _chunk("c1", "TLS 1.2 required.", "d1", {"published_at": "2022-06-01"}),
        _chunk("c2", "TLS 1.3 required.", "d2", {"published_at": "2026-03-01"}),
    ]
    run = _run(
        chunks,
        query="Which TLS version is required?",
        answer="TLS 1.2 is required.",
    )
    result = StaleRetrievalAnalyzer().analyze(run)
    assert result.status == "fail"
    assert result.failure_type == FailureType.STALE_RETRIEVAL


def test_relative_recency_skips_when_only_one_chunk_has_temporal_metadata() -> None:
    """Need ≥2 chunks with temporal data to make a relative claim."""
    chunks = [
        _chunk("c1", "Rate is X.", "d1", {"effective_date": "2025-01-01"}),
        _chunk("c2", "Rate is Y.", "d2", {}),
    ]
    run = _run(chunks, query="What is X?", answer="X is Y.")
    result = StaleRetrievalAnalyzer().analyze(run)
    # Fall through to legacy path which skips without corpus metadata.
    assert result.failure_type != FailureType.STALE_RETRIEVAL


def test_relative_recency_threshold_blocks_trivial_date_deltas() -> None:
    """A 1-day delta should not flag staleness; threshold is days-scale."""
    chunks = [
        _chunk("c1", "Policy version A.", "d1", {"effective_date": "2026-01-01"}),
        _chunk("c2", "Policy version B.", "d2", {"effective_date": "2026-01-02"}),
    ]
    run = _run(chunks, query="What is the policy?", answer="The policy is version A.")
    result = StaleRetrievalAnalyzer().analyze(run)
    # 1-day delta below default 30-day min — must not fire.
    assert result.failure_type != FailureType.STALE_RETRIEVAL


def test_legacy_skip_preserved_when_no_metadata_anywhere() -> None:
    """The pre-existing skip contract must not regress."""
    chunks = [_chunk("c1", "text", "d1", {})]
    run = _run(chunks, query="q", answer="a")
    result = StaleRetrievalAnalyzer().analyze(run)
    assert result.status == "skip"
    assert result.evidence == ["no corpus metadata available"]
