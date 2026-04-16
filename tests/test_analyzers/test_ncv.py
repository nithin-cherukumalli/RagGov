"""Tests for NCV pipeline verifier."""

from __future__ import annotations

import json

from raggov.analyzers.verification.ncv import NCVNode, NCVPipelineVerifier
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


def run_with_chunks(
    chunks: list[RetrievedChunk],
    *,
    query: str = "What is Rule 5?",
    answer: str = "Rule 5 requires notice.",
) -> RAGRun:
    return RAGRun(
        query=query,
        retrieved_chunks=chunks,
        final_answer=answer,
    )


def test_all_nodes_pass_pipeline_health_is_one() -> None:
    analyzer = NCVPipelineVerifier()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "The district has 12 vacancies under Rule 5.", 0.82),
                chunk("chunk-2", "Rule 5 records 12 vacancies for the district.", 0.79),
            ],
            query="How many vacancies are there?",
            answer="There are 12 vacancies.",
        )
    )

    assert result.status == "pass"
    assert result.failure_type is None
    assert result.score == 1.0

    report = json.loads(result.evidence[0])
    assert report["pipeline_health_score"] == 1.0
    assert report["first_failing_node"] is None


def test_retrieval_quality_fail_stops_in_fail_fast_mode() -> None:
    analyzer = NCVPipelineVerifier()
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice.", 0.31),
                chunk("chunk-2", "Rule 5 also references appeals.", 0.28),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.RETRIEVAL_QUALITY.value
    assert [node["node"] for node in report["node_results"]] == [
        NCVNode.QUERY_UNDERSTANDING.value,
        NCVNode.RETRIEVAL_QUALITY.value,
    ]


def test_claim_grounding_fail_maps_to_unsupported_claim() -> None:
    analyzer = NCVPipelineVerifier({"fail_fast": False})
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice before action.", 0.88),
            ],
            query="What is Rule 5?",
            answer="Rule 5 creates a pension benefit for teachers.",
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GROUNDING

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.CLAIM_GROUNDING.value


def test_duplicate_chunks_fail_context_assembly() -> None:
    analyzer = NCVPipelineVerifier({"fail_fast": False})
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice before action in Hyderabad.", 0.84),
                chunk("chunk-2", "Rule 5 requires notice before action in Hyderabad.", 0.83),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INCONSISTENT_CHUNKS
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.CONTEXT_ASSEMBLY.value


def test_answer_completeness_fails_for_missing_number() -> None:
    analyzer = NCVPipelineVerifier({"fail_fast": False})
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "There are several vacancies in the district.", 0.87),
            ],
            query="How many vacancies are there?",
            answer="There are several vacancies.",
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GROUNDING

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.ANSWER_COMPLETENESS.value


def test_empty_chunks_record_two_failures_before_stopping() -> None:
    analyzer = NCVPipelineVerifier()
    result = analyzer.analyze(run_with_chunks([], answer="No answer."))

    assert result.status == "fail"
    assert result.failure_type == FailureType.SCOPE_VIOLATION
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert [node["node"] for node in report["node_results"]] == [
        NCVNode.QUERY_UNDERSTANDING.value,
        NCVNode.RETRIEVAL_QUALITY.value,
    ]
    assert all(node["status"] == "fail" for node in report["node_results"])


def test_uncertain_path_returns_warn_with_report() -> None:
    analyzer = NCVPipelineVerifier({"fail_fast": False})
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice before action.", None),
            ],
            query="Explain Rule 5",
            answer="Rule 5 requires notice before action.",
        )
    )

    assert result.status == "warn"
    assert result.failure_type is None

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] is None
    assert any(node["status"] == "uncertain" for node in report["node_results"])
