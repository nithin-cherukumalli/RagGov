from __future__ import annotations

import json

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.attribution.pinpoint_context import (
    get_pinpoint_findings_for_a2p,
    summarize_pinpoint_findings_for_a2p,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.ncv import (
    NCVCalibrationStatus,
    NCVEvidenceSignal,
    NCVMethodType,
    NCVNode,
    NCVNodeResult,
    NCVNodeStatus,
    NCVReport,
)
from raggov.models.pinpoint import PinpointEvidence, PinpointFinding, PinpointLocation
from raggov.models.run import RAGRun


def _ncv_report(first_failing_node: NCVNode = NCVNode.RETRIEVAL_COVERAGE) -> NCVReport:
    node_result = NCVNodeResult(
        node=first_failing_node,
        status=NCVNodeStatus.FAIL,
        primary_reason="Retrieval miss confirmed.",
        evidence_signals=[
            NCVEvidenceSignal(
                signal_name="retrieval_primary_failure",
                value="retrieval_miss",
                source_report="RetrievalDiagnosisReport",
                source_ids=["chunk-1"],
                interpretation="Coverage gap detected.",
            )
        ],
        affected_claim_ids=["claim-1"],
        affected_chunk_ids=["chunk-1"],
        affected_doc_ids=["doc-1"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        fallback_used=False,
        limitations=["Heuristic NCV output."],
    )
    return NCVReport(
        run_id="run-pinpoint",
        node_results=[node_result],
        first_failing_node=first_failing_node,
        bottleneck_description="Pipeline fails at retrieval_coverage: retrieval miss.",
        downstream_failure_chain=[first_failing_node],
        evidence_reports_used=["RetrievalDiagnosisReport"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        recommended_for_gating=False,
        limitations=["Uncalibrated NCV report."],
    )


def _run_with_ncv_report(report: dict | NCVReport | None) -> RAGRun:
    metadata = {"reports": {}}
    if report is not None:
        metadata["reports"]["ncv_report"] = (
            report if isinstance(report, dict) else report.model_dump(mode="json")
        )
    return RAGRun(
        run_id="run-pinpoint",
        query="What is the policy?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Policy text.",
                source_doc_id="doc-1",
                score=0.9,
            )
        ],
        final_answer="Policy answer.",
        metadata=metadata,
    )


def test_get_pinpoint_findings_for_a2p_extracts_from_run_metadata_reports() -> None:
    run = _run_with_ncv_report(_ncv_report())

    findings = get_pinpoint_findings_for_a2p(run)

    assert len(findings) == 1
    assert findings[0].location.ncv_node == "retrieval_coverage"
    assert findings[0].recommended_for_gating is False


def test_summarize_pinpoint_findings_for_a2p_produces_compact_summary() -> None:
    finding = PinpointFinding(
        finding_id="finding-1",
        location=PinpointLocation(
            location_id="loc-1",
            ncv_node="citation_support",
            pipeline_stage="grounding",
            failure_type="CITATION_MISMATCH",
            claim_ids=["claim-1"],
            chunk_ids=["chunk-1"],
            doc_ids=["doc-1"],
            localization_method="ncv_first_failing_node_v1",
            method_type="evidence_aggregation",
            calibration_status="uncalibrated",
            recommended_for_gating=False,
            limitations=["Uncalibrated pinpoint."],
        ),
        evidence_for=[
            PinpointEvidence(
                signal_name="citation_faithfulness_issues",
                value=1,
                source_report="CitationFaithfulnessReport",
                interpretation="Citation mismatch confirmed.",
                affected_claim_ids=["claim-1"],
                affected_chunk_ids=["chunk-1"],
                affected_doc_ids=["doc-1"],
                method_type="evidence_aggregation",
                calibration_status="uncalibrated",
                limitations=[],
            )
        ],
        missing_evidence=["supporting citation"],
        fallback_heuristics_used=["native_citation_verifier"],
        calibration_status="uncalibrated",
        recommended_for_gating=False,
    )

    summary = summarize_pinpoint_findings_for_a2p([finding])

    assert summary["pinpoint_available"] is True
    assert summary["primary_ncv_node"] == "citation_support"
    assert summary["calibration_status"] == "uncalibrated"
    assert summary["recommended_for_gating"] is False
    assert summary["affected_claim_ids"] == ["claim-1"]
    assert summary["affected_chunk_ids"] == ["chunk-1"]
    assert summary["affected_doc_ids"] == ["doc-1"]


def test_a2p_with_ncv_pinpoint_context_attaches_summary_to_result() -> None:
    run = _run_with_ncv_report(_ncv_report(NCVNode.CITATION_SUPPORT))
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        claim_results=[
            ClaimResult(
                claim_text="Policy answer.",
                label="unsupported",
                candidate_chunk_ids=["chunk-1"],
            )
        ],
    )
    analyzer = A2PAttributionAnalyzer({"prior_results": [prior_result]})

    result = analyzer.analyze(run)

    assert result.status == "fail"
    line = next(
        item for item in result.evidence if item.startswith("a2p_pinpoint_context:")
    )
    payload = json.loads(line.split(":", 1)[1])
    assert payload["primary_ncv_node"] == "citation_support"
    assert payload["recommended_for_gating"] is False
    assert payload["calibration_status"] == "uncalibrated"
    assert result.claim_attributions is not None
    assert any("primary_ncv_node=citation_support" in item for item in result.claim_attributions[0].evidence)


def test_a2p_without_ncv_report_remains_backward_compatible() -> None:
    run = _run_with_ncv_report(None)
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        claim_results=[
            ClaimResult(
                claim_text="Policy answer.",
                label="unsupported",
                candidate_chunk_ids=["chunk-1"],
            )
        ],
    )
    analyzer = A2PAttributionAnalyzer({"prior_results": [prior_result]})

    result = analyzer.analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GROUNDING
    assert not any(item.startswith("a2p_pinpoint_context:") for item in result.evidence)


def test_malformed_ncv_report_does_not_crash_a2p() -> None:
    run = _run_with_ncv_report({"malformed": True})
    prior_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        claim_results=[
            ClaimResult(
                claim_text="Policy answer.",
                label="unsupported",
                candidate_chunk_ids=["chunk-1"],
            )
        ],
    )
    analyzer = A2PAttributionAnalyzer({"prior_results": [prior_result]})

    findings = get_pinpoint_findings_for_a2p(run)
    result = analyzer.analyze(run)

    assert findings == []
    assert result.status == "fail"
    assert not any(item.startswith("a2p_pinpoint_context:") for item in result.evidence)
