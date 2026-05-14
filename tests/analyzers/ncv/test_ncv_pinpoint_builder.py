"""Tests for the NCV-to-pinpoint builder.

Verifies that build_pinpoint_findings_from_ncv_report correctly converts
NCVReport objects and plain dicts into structured PinpointFinding objects
without altering NCVPipelineVerifier or DiagnosisEngine behavior.
"""

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
from raggov.pinpoint import build_pinpoint_findings_from_ncv_report


def _make_signal(
    signal_name: str = "test_signal",
    value: object = "does_not_support",
    source_report: str = "CitationFaithfulnessReport",
    interpretation: str = "Citation does not support claim.",
    limitation: str | None = None,
) -> NCVEvidenceSignal:
    return NCVEvidenceSignal(
        signal_name=signal_name,
        value=value,
        source_report=source_report,
        source_ids=[],
        interpretation=interpretation,
        limitation=limitation,
    )


def _make_node_result(
    node: NCVNode,
    status: NCVNodeStatus = NCVNodeStatus.FAIL,
    signals: list[NCVEvidenceSignal] | None = None,
    affected_claim_ids: list[str] | None = None,
    affected_chunk_ids: list[str] | None = None,
    affected_doc_ids: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    fallback_used: bool = False,
    limitations: list[str] | None = None,
    node_score: float | None = None,
) -> NCVNodeResult:
    return NCVNodeResult(
        node=node,
        status=status,
        primary_reason=f"{node.value} test failure",
        evidence_signals=signals or [],
        affected_claim_ids=affected_claim_ids or [],
        affected_chunk_ids=affected_chunk_ids or [],
        affected_doc_ids=affected_doc_ids or [],
        missing_evidence=missing_evidence or [],
        fallback_used=fallback_used,
        limitations=limitations or [],
        node_score=node_score,
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )


def _make_report(
    first_failing_node: NCVNode,
    node_result: NCVNodeResult,
    fallback_heuristics_used: list[str] | None = None,
) -> NCVReport:
    return NCVReport(
        run_id="run-test",
        node_results=[node_result],
        first_failing_node=first_failing_node,
        bottleneck_description=f"Pipeline fails at {first_failing_node.value}.",
        evidence_reports_used=[],
        fallback_heuristics_used=fallback_heuristics_used or [],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )


class TestCitationSupportBuilder:
    """Builder creates one finding for citation_support first_failing_node."""

    def test_returns_one_finding(self):
        signal = _make_signal(
            signal_name="citation_does_not_support",
            value="does_not_support",
            source_report="CitationFaithfulnessReport",
            interpretation="Citation does not support the claim.",
        )
        node = _make_node_result(
            node=NCVNode.CITATION_SUPPORT,
            signals=[signal],
            affected_claim_ids=["claim-1"],
        )
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert len(findings) == 1

    def test_location_ncv_node_matches(self):
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[_make_signal()])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].location.ncv_node == NCVNode.CITATION_SUPPORT.value

    def test_recommended_for_gating_is_false(self):
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[_make_signal()])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].recommended_for_gating is False
        assert findings[0].location.recommended_for_gating is False

    def test_evidence_for_is_populated(self):
        signal = _make_signal(signal_name="citation_signal", value="unsupported")
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[signal])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert len(findings[0].evidence_for) == 1
        assert findings[0].evidence_for[0].signal_name == "citation_signal"

    def test_calibrated_confidence_is_none(self):
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[_make_signal()])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].calibrated_confidence is None

    def test_accepts_dict_input(self):
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[_make_signal()])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)
        report_dict = report.model_dump(mode="json")

        findings = build_pinpoint_findings_from_ncv_report(report_dict)

        assert len(findings) == 1
        assert findings[0].location.ncv_node == "citation_support"
        assert findings[0].calibrated_confidence is None


class TestRetrievalCoverageBuilder:
    """Builder creates one finding for retrieval_coverage first_failing_node."""

    def test_location_ncv_node_matches(self):
        signal = _make_signal(
            signal_name="retrieval_primary_failure",
            value="retrieval_miss",
            source_report="RetrievalDiagnosisReport",
            interpretation="Retrieval diagnosis reports retrieval miss.",
        )
        node = _make_node_result(
            NCVNode.RETRIEVAL_COVERAGE,
            signals=[signal],
            affected_claim_ids=["claim-1"],
        )
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert len(findings) == 1
        assert findings[0].location.ncv_node == NCVNode.RETRIEVAL_COVERAGE.value

    def test_missing_evidence_preserved(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_COVERAGE,
            signals=[_make_signal()],
            missing_evidence=["retrieval_diagnosis_report", "embedding_profile"],
        )
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert "retrieval_diagnosis_report" in findings[0].missing_evidence
        assert "embedding_profile" in findings[0].missing_evidence

    def test_human_review_recommended_is_true_when_uncalibrated(self):
        node = _make_node_result(NCVNode.RETRIEVAL_COVERAGE, signals=[_make_signal()])
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].human_review_recommended is True

    def test_human_review_true_when_missing_evidence_present(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_COVERAGE,
            signals=[_make_signal()],
            missing_evidence=["retrieval_diagnosis_report"],
        )
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].human_review_recommended is True

    def test_affected_claim_ids_in_location(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_COVERAGE,
            signals=[_make_signal()],
            affected_claim_ids=["claim-1", "claim-2"],
        )
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].location.claim_ids == ["claim-1", "claim-2"]


class TestFallbackPreservation:
    """Fallback heuristics from NCVReport and node.fallback_used flow into finding."""

    def test_report_fallback_heuristics_preserved(self):
        node = _make_node_result(NCVNode.RETRIEVAL_PRECISION, signals=[_make_signal()])
        report = _make_report(
            NCVNode.RETRIEVAL_PRECISION,
            node,
            fallback_heuristics_used=["mean_retrieval_score_threshold"],
        )

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert "mean_retrieval_score_threshold" in findings[0].fallback_heuristics_used

    def test_node_fallback_used_adds_label(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_PRECISION,
            signals=[_make_signal()],
            fallback_used=True,
        )
        report = _make_report(NCVNode.RETRIEVAL_PRECISION, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        fallback_label = "ncv_node_retrieval_precision_fallback"
        assert fallback_label in findings[0].fallback_heuristics_used

    def test_combined_report_and_node_fallbacks_merged(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_PRECISION,
            signals=[_make_signal()],
            fallback_used=True,
        )
        report = _make_report(
            NCVNode.RETRIEVAL_PRECISION,
            node,
            fallback_heuristics_used=["mean_retrieval_score_threshold"],
        )

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert "mean_retrieval_score_threshold" in findings[0].fallback_heuristics_used
        assert "ncv_node_retrieval_precision_fallback" in findings[0].fallback_heuristics_used

    def test_recommended_for_gating_remains_false_with_fallbacks(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_PRECISION,
            signals=[_make_signal()],
            fallback_used=True,
        )
        report = _make_report(
            NCVNode.RETRIEVAL_PRECISION,
            node,
            fallback_heuristics_used=["mean_retrieval_score_threshold"],
        )

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].recommended_for_gating is False


class TestNoFailingNode:
    """Builder returns empty list when first_failing_node is None."""

    def test_empty_list_on_no_first_failing_node(self):
        report = NCVReport(
            run_id="run-clean",
            node_results=[],
            first_failing_node=None,
            bottleneck_description="No failure detected.",
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        )

        assert build_pinpoint_findings_from_ncv_report(report) == []

    def test_empty_list_on_none_dict(self):
        report_dict = {
            "run_id": "run-clean",
            "node_results": [],
            "first_failing_node": None,
            "bottleneck_description": "No failure.",
            "method_type": "evidence_aggregation",
            "calibration_status": "uncalibrated",
            "recommended_for_gating": False,
            "limitations": [],
        }

        assert build_pinpoint_findings_from_ncv_report(report_dict) == []

    def test_empty_list_on_invalid_dict(self):
        assert build_pinpoint_findings_from_ncv_report({"invalid": "data"}) == []


class TestEvidenceSignalMapping:
    """Evidence signal fields are faithfully mapped into PinpointEvidence."""

    def test_signal_name_and_value_preserved(self):
        signal = _make_signal(signal_name="retrieval_miss_flag", value=True)
        node = _make_node_result(NCVNode.RETRIEVAL_COVERAGE, signals=[signal])
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)
        ev = findings[0].evidence_for[0]

        assert ev.signal_name == "retrieval_miss_flag"
        assert ev.value is True

    def test_signal_source_report_preserved(self):
        signal = _make_signal(source_report="RetrievalDiagnosisReport")
        node = _make_node_result(NCVNode.RETRIEVAL_COVERAGE, signals=[signal])
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].evidence_for[0].source_report == "RetrievalDiagnosisReport"

    def test_signal_none_source_report_uses_fallback(self):
        signal = NCVEvidenceSignal(
            signal_name="fallback_signal",
            value="miss",
            source_report=None,
            source_ids=[],
            interpretation="Source report unavailable.",
        )
        node = _make_node_result(NCVNode.RETRIEVAL_COVERAGE, signals=[signal])
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].evidence_for[0].source_report == "ncv_pipeline_verifier"

    def test_signal_limitation_mapped_to_limitations_list(self):
        signal = _make_signal(limitation="uncalibrated heuristic")
        node = _make_node_result(NCVNode.RETRIEVAL_COVERAGE, signals=[signal])
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert "uncalibrated heuristic" in findings[0].evidence_for[0].limitations

    def test_heuristic_score_preserved_from_node(self):
        node = _make_node_result(
            NCVNode.RETRIEVAL_COVERAGE,
            signals=[_make_signal()],
            node_score=0.25,
        )
        report = _make_report(NCVNode.RETRIEVAL_COVERAGE, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].heuristic_score == 0.25

    def test_calibration_status_is_uncalibrated(self):
        node = _make_node_result(NCVNode.CITATION_SUPPORT, signals=[_make_signal()])
        report = _make_report(NCVNode.CITATION_SUPPORT, node)

        findings = build_pinpoint_findings_from_ncv_report(report)

        assert findings[0].calibration_status == "uncalibrated"
        assert findings[0].evidence_for[0].calibration_status == "uncalibrated"
