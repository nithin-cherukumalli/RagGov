"""Tests for NCV pipeline verifier."""

from __future__ import annotations

import json

from raggov.analyzers.verification.ncv import NCVPipelineVerifier
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
    SecurityRisk,
    SufficiencyResult,
)
from raggov.models.ncv import NCVNode
from raggov.models.retrieval_diagnosis import (
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalFailureType,
)
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    EvidenceRole,
    QueryRelevanceLabel,
    RetrievalEvidenceProfile,
)
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


def _node_payload(report: dict[str, object], node: NCVNode) -> dict[str, object]:
    return next(item for item in report["node_results"] if item["node"] == node.value)


def test_all_nodes_pass_pipeline_health_is_one() -> None:
    diagnosis = RetrievalDiagnosisReport(
        run_id="run-ncv-pass",
        primary_failure_type=RetrievalFailureType.NO_CLEAR_RETRIEVAL_FAILURE,
        recommended_fix="No retrieval fix indicated.",
        method_type=RetrievalDiagnosisMethodType.PRACTICAL_APPROXIMATION,
        calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
    )
    prior = AnalyzerResult(
        analyzer_name="RetrievalDiagnosisAnalyzerV0",
        status="pass",
        retrieval_diagnosis_report=diagnosis,
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior]})
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
    assert report["recommended_for_gating"] is False
    assert all("method_type" in node for node in report["node_results"])
    assert all("calibration_status" in node for node in report["node_results"])


def test_retrieval_quality_fail_stops_in_fail_fast_mode() -> None:
    analyzer = NCVPipelineVerifier({"allow_retrieval_precision_fallback": True})
    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice.", 0.31),
                chunk("chunk-2", "Rule 5 also references appeals.", 0.28),
            ]
        )
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.RETRIEVAL_ANOMALY
    assert result.stage == FailureStage.RETRIEVAL

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.RETRIEVAL_PRECISION.value
    assert [node["node"] for node in report["node_results"]] == [
        NCVNode.QUERY_UNDERSTANDING.value,
        NCVNode.PARSER_VALIDITY.value,
        NCVNode.RETRIEVAL_COVERAGE.value,
        NCVNode.RETRIEVAL_PRECISION.value,
    ]
    assert "mean_retrieval_score_threshold" in report["fallback_heuristics_used"]


def test_retrieval_anomaly_alone_does_not_trigger_security_risk() -> None:
    prior = AnalyzerResult(
        analyzer_name="RetrievalAnomalyAnalyzer",
        status="warn",
        failure_type=FailureType.RETRIEVAL_ANOMALY,
        stage=FailureStage.SECURITY,
        evidence=["score cliff between chunk-1 and chunk-2"],
    )
    analyzer = NCVPipelineVerifier(
        {"prior_results": [prior], "fail_fast": False, "allow_retrieval_precision_fallback": True}
    )

    result = analyzer.analyze(
        run_with_chunks(
            [
                chunk("chunk-1", "Rule 5 requires notice.", 0.31),
                chunk("chunk-2", "Rule 5 requires review.", 0.28),
            ]
        )
    )

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] != NCVNode.SECURITY_RISK.value
    retrieval_node = _node_payload(report, NCVNode.RETRIEVAL_PRECISION)
    assert retrieval_node["status"] in {"warn", "fail"}
    security_node = _node_payload(report, NCVNode.SECURITY_RISK)
    assert security_node["status"] == "skip"


def test_prompt_injection_still_triggers_security_risk() -> None:
    prior = AnalyzerResult(
        analyzer_name="PromptInjectionAnalyzer",
        status="fail",
        failure_type=FailureType.PROMPT_INJECTION,
        stage=FailureStage.SECURITY,
        security_risk=SecurityRisk.HIGH,
        remediation="Ignore hostile instructions in retrieved text.",
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks(
            [chunk("chunk-1", "Rule 5 requires notice before action.", 0.88)],
            query="What does Rule 5 require?",
            answer="Rule 5 requires notice before action.",
        )
    )

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.SECURITY_RISK.value
    security_node = _node_payload(report, NCVNode.SECURITY_RISK)
    assert security_node["status"] == "fail"
    assert "prompt injection" in security_node["primary_reason"].lower()


def test_suspicious_chunk_still_triggers_security_risk() -> None:
    prior = AnalyzerResult(
        analyzer_name="SuspiciousChunkAnalyzer",
        status="warn",
        failure_type=FailureType.SUSPICIOUS_CHUNK,
        stage=FailureStage.SECURITY,
        security_risk=SecurityRisk.MEDIUM,
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks(
            [chunk("chunk-1", "Rule 5 requires notice before action.", 0.88)],
            query="What does Rule 5 require?",
            answer="Rule 5 requires notice before action.",
        )
    )

    report = json.loads(result.evidence[0])
    security_node = _node_payload(report, NCVNode.SECURITY_RISK)
    assert security_node["status"] == "warn"
    assert "suspicious" in security_node["primary_reason"].lower()


def test_privacy_violation_still_triggers_security_risk() -> None:
    prior = AnalyzerResult(
        analyzer_name="PrivacyAnalyzer",
        status="fail",
        failure_type=FailureType.PRIVACY_VIOLATION,
        stage=FailureStage.SECURITY,
        security_risk=SecurityRisk.HIGH,
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks(
            [chunk("chunk-1", "Rule 5 requires notice before action.", 0.88)],
            query="What does Rule 5 require?",
            answer="Rule 5 requires notice before action.",
        )
    )

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == NCVNode.SECURITY_RISK.value
    security_node = _node_payload(report, NCVNode.SECURITY_RISK)
    assert security_node["status"] == "fail"
    assert "privacy" in security_node["primary_reason"].lower()


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
    assert report["first_failing_node"] == NCVNode.CLAIM_SUPPORT.value


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
    assert result.failure_type == FailureType.GENERATION_IGNORE
    assert result.stage == FailureStage.GENERATION

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
        NCVNode.PARSER_VALIDITY.value,
        NCVNode.RETRIEVAL_COVERAGE.value,
    ]
    assert report["node_results"][0]["status"] == "fail"
    assert report["node_results"][1]["status"] == "skip"
    assert report["node_results"][2]["status"] == "fail"


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
    assert "retrieval_diagnosis_report" in report["missing_reports"]


def test_retrieval_diagnosis_report_drives_retrieval_coverage_failure() -> None:
    diagnosis = RetrievalDiagnosisReport(
        run_id="run-ncv",
        primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
        affected_claim_ids=["claim-1"],
        recommended_fix="Expand retrieval depth and inspect query routing.",
        method_type=RetrievalDiagnosisMethodType.PRACTICAL_APPROXIMATION,
        calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
    )
    prior = AnalyzerResult(
        analyzer_name="RetrievalDiagnosisAnalyzerV0",
        status="fail",
        retrieval_diagnosis_report=diagnosis,
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Rule 5 requires notice.", 0.88)])
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.RETRIEVAL
    assert result.remediation == "Expand retrieval depth and inspect query routing."

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == "retrieval_coverage"
    assert report["method_type"] == "evidence_aggregation"
    assert "RetrievalDiagnosisReport" in report["evidence_reports_used"]
    node = next(node for node in report["node_results"] if node["node"] == "retrieval_coverage")
    assert node["affected_claim_ids"] == ["claim-1"]
    assert node["evidence_signals"]


def test_query_relevance_profile_prevents_lexical_query_fallback_failure() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-ncv",
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="chunk-1",
                query_relevance_label=QueryRelevanceLabel.RELEVANT,
                evidence_role=EvidenceRole.NECESSARY_SUPPORT,
            )
        ],
    )
    analyzer = NCVPipelineVerifier({"fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks(
            [chunk("chunk-1", "Administrative notice is required before action.", 0.88)],
            query="Explain Rule 5",
            answer="Notice is required before action.",
        ).model_copy(update={"retrieval_evidence_profile": profile})
    )

    report = json.loads(result.evidence[0])
    query_node = next(node for node in report["node_results"] if node["node"] == "query_understanding")
    assert query_node["status"] == "pass"
    assert query_node["method_type"] == "evidence_aggregation"
    assert "query_term_overlap" not in report["fallback_heuristics_used"]


def test_claim_grounding_prior_result_drives_claim_support_failure() -> None:
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        claim_results=[
            ClaimResult(
                claim_text="Rule 5 creates a pension benefit.",
                label="unsupported",
                candidate_chunk_ids=["chunk-1"],
            )
        ],
    )
    analyzer = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})

    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Rule 5 requires notice.", 0.88)])
    )

    report = json.loads(result.evidence[0])
    node = next(node for node in report["node_results"] if node["node"] == "claim_support")
    assert node["status"] == "fail"
    assert node["affected_claim_ids"] == ["claim-1"]
    assert node["affected_chunk_ids"] == ["chunk-1"]
    assert node["method_type"] == "evidence_aggregation"


def test_insufficient_sufficiency_and_unsupported_claims_fail_coverage() -> None:
    sufficiency = AnalyzerResult(
        analyzer_name="ClaimAwareSufficiencyAnalyzer",
        status="fail",
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            sufficiency_label="insufficient",
            affected_claims=["claim-1"],
            missing_evidence=["missing statutory exception"],
            method="test",
        ),
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        claim_results=[ClaimResult(claim_text="Unsupported claim.", label="unsupported")],
    )
    analyzer = NCVPipelineVerifier({"prior_results": [sufficiency, grounding]})

    result = analyzer.analyze(
        run_with_chunks([chunk("chunk-1", "Rule 5 requires notice.", 0.88)])
    )

    report = json.loads(result.evidence[0])
    assert report["first_failing_node"] == "retrieval_coverage"
    node = next(node for node in report["node_results"] if node["node"] == "retrieval_coverage")
    assert node["missing_evidence"] == ["missing statutory exception"]
