"""Tests for NCV evidence aggregation data models."""

from __future__ import annotations

from raggov.models.ncv import (
    NCVCalibrationStatus,
    NCVEvidenceSignal,
    NCVMethodType,
    NCVNode,
    NCVNodeResult,
    NCVNodeStatus,
    NCVReport,
)


def test_ncv_enum_values() -> None:
    assert NCVNode.QUERY_UNDERSTANDING == "query_understanding"
    assert NCVNode.PARSER_VALIDITY == "parser_validity"
    assert NCVNode.RETRIEVAL_COVERAGE == "retrieval_coverage"
    assert NCVNode.RETRIEVAL_PRECISION == "retrieval_precision"
    assert NCVNode.CONTEXT_ASSEMBLY == "context_assembly"
    assert NCVNode.VERSION_VALIDITY == "version_validity"
    assert NCVNode.CLAIM_SUPPORT == "claim_support"
    assert NCVNode.CITATION_SUPPORT == "citation_support"
    assert NCVNode.ANSWER_COMPLETENESS == "answer_completeness"
    assert NCVNode.SECURITY_RISK == "security_risk"
    assert NCVNodeStatus.UNCERTAIN == "uncertain"
    assert NCVMethodType.EVIDENCE_AGGREGATION == "evidence_aggregation"
    assert NCVCalibrationStatus.UNCALIBRATED == "uncalibrated"


def test_ncv_report_serializes_required_metadata() -> None:
    signal = NCVEvidenceSignal(
        signal_name="retrieval_primary_failure",
        value="retrieval_miss",
        source_report="RetrievalDiagnosisReport",
        source_ids=["claim-1"],
        interpretation="Retrieval diagnosis attributes unsupported claim to retrieval miss.",
    )
    node = NCVNodeResult(
        node=NCVNode.RETRIEVAL_COVERAGE,
        status=NCVNodeStatus.FAIL,
        primary_reason="Retrieval diagnosis reports retrieval miss.",
        evidence_signals=[signal],
        affected_claim_ids=["claim-1"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )
    report = NCVReport(
        run_id="run-1",
        node_results=[node],
        first_failing_node=NCVNode.RETRIEVAL_COVERAGE,
        pipeline_health_score=0.0,
        bottleneck_description="Pipeline fails at retrieval_coverage.",
        evidence_reports_used=["RetrievalDiagnosisReport"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )

    dumped = report.model_dump(mode="json")

    assert dumped["first_failing_node"] == "retrieval_coverage"
    assert dumped["node_results"][0]["method_type"] == "evidence_aggregation"
    assert dumped["node_results"][0]["calibration_status"] == "uncalibrated"
    assert dumped["recommended_for_gating"] is False
