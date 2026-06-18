"""Task 16: broad-permission claim vs explicit restriction is a CONTRADICTED_CLAIM."""
from __future__ import annotations

from raggov import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureType
from raggov.models.run import RAGRun


def _run(answer: str, chunk_text: str) -> RAGRun:
    return RAGRun(
        query="What is the current policy?",
        retrieved_chunks=[RetrievedChunk(chunk_id="c1", text=chunk_text, source_doc_id="d1", score=0.9)],
        final_answer=answer,
        cited_doc_ids=["d1"],
    )


def test_broad_permission_vs_restriction_is_contradicted() -> None:
    d = diagnose(_run("The policy says you can wear anything you want.", "Policy: No blue shirts."))
    assert d.primary_failure == FailureType.CONTRADICTED_CLAIM


def test_permission_claim_consistent_with_evidence_not_contradicted() -> None:
    # Precision guard: a claim consistent with the evidence must NOT be contradiction.
    d = diagnose(_run("The policy says blue shirts are not allowed.", "Policy: No blue shirts."))
    assert d.primary_failure != FailureType.CONTRADICTED_CLAIM
