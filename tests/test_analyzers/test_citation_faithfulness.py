"""Tests for citation faithfulness probing."""

from __future__ import annotations

from pathlib import Path

from raggov.analyzers.grounding.citation_faithfulness import CitationFaithfulnessProbe
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def chunk(
    chunk_id: str,
    text: str,
    *,
    source_doc_id: str | None = None,
    score: float | None = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id or f"doc-{chunk_id}",
        score=score,
    )


def run_with_answer(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is the policy?",
    cited_doc_ids: list[str] | None = None,
    answer_confidence: float | None = 0.8,
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer=answer,
        cited_doc_ids=cited_doc_ids or [],
        answer_confidence=answer_confidence,
    )


def test_answer_with_rare_anchor_terms_passes() -> None:
    run = run_with_answer(
        "The policy requires xenolith calibration and a fluxgate reading before approval.",
        [
            chunk(
                "chunk-1",
                "The policy requires xenolith calibration and a fluxgate reading before approval.",
                source_doc_id="doc-1",
            ),
            chunk(
                "chunk-2",
                "General policy requires approval before processing and routine verification.",
                source_doc_id="doc-2",
            ),
        ],
        cited_doc_ids=["doc-1"],
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None
    assert result.citation_probe_results is not None


def test_cited_doc_without_unique_term_use_fails() -> None:
    run = run_with_answer(
        "The policy requires approval before processing.",
        [
            chunk(
                "chunk-1",
                "The policy requires xenolith calibration and a fluxgate reading before approval.",
                source_doc_id="doc-1",
            ),
            chunk(
                "chunk-2",
                "General policy requires approval before processing and routine verification.",
                source_doc_id="doc-2",
            ),
            chunk(
                "chunk-3",
                "Another cited page mentions zephyrite shielding and quorite handling.",
                source_doc_id="doc-3",
            ),
        ],
        cited_doc_ids=["doc-1", "doc-3"],
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION
    assert result.stage == FailureStage.GROUNDING


def test_empty_citations_with_confident_answer_fails() -> None:
    run = run_with_answer(
        "The policy allows refunds within thirty days.",
        [chunk("chunk-1", "Refund policy text.", source_doc_id="doc-1")],
        cited_doc_ids=[],
        answer_confidence=0.91,
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION
    assert "no citations provided" in " ".join(result.evidence).lower()


def test_claim_coverage_gap_fails_when_half_the_claims_are_uncited() -> None:
    run = run_with_answer(
        "Alpha protocol requires a xenolith seal. Appeals are decided by a lunar board.",
        [
            chunk(
                "chunk-1",
                "Alpha protocol requires a xenolith seal before dispatch.",
                source_doc_id="doc-1",
            ),
        ],
        cited_doc_ids=["doc-1"],
    )

    result = CitationFaithfulnessProbe().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION


def test_integration_fixture_citation_mismatch_triggers_probe() -> None:
    fixture_run = RAGRun.model_validate_json((FIXTURES / "citation_mismatch.json").read_text())

    result = CitationFaithfulnessProbe().analyze(fixture_run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.POST_RATIONALIZED_CITATION
    assert result.citation_probe_results is not None
