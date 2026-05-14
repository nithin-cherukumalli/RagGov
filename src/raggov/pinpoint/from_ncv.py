"""Build structured PinpointFinding objects from an NCVReport.

This builder is a read-only adapter: it does not alter NCVPipelineVerifier
behavior, does not change first_failing_node logic, and does not introduce
calibrated confidence.  All findings are explicitly marked uncalibrated and
not recommended for gating.
"""

from __future__ import annotations

from raggov.models.ncv import NCVCalibrationStatus, NCVReport
from raggov.models.pinpoint import PinpointEvidence, PinpointFinding, PinpointLocation


def build_pinpoint_findings_from_ncv_report(
    ncv_report: NCVReport | dict,
) -> list[PinpointFinding]:
    """Return a list of PinpointFinding objects derived from an NCVReport.

    Returns an empty list when first_failing_node is None (no failure detected)
    or when the report cannot be parsed.

    Args:
        ncv_report: An NCVReport model instance or a plain dict (e.g. from
            JSON-deserializing result.evidence[0]).

    Returns:
        A list containing at most one PinpointFinding for the first failing node.
    """
    report = _coerce_to_report(ncv_report)
    if report is None or report.first_failing_node is None:
        return []

    target_node = report.first_failing_node
    node_result = next(
        (n for n in report.node_results if n.node == target_node),
        None,
    )
    if node_result is None:
        return []

    method_type_str = node_result.method_type.value
    calibration_str = node_result.calibration_status.value

    location_limitations = list(node_result.limitations)
    if report.priority_policy_decision:
        location_limitations.append(
            f"priority_policy_v1: {report.priority_policy_decision.get('reason', 'applied')}"
        )

    location = PinpointLocation(
        location_id=f"ncv_{target_node.value}_{report.run_id}",
        ncv_node=target_node.value,
        localization_method="ncv_first_failing_node_v1",
        method_type=method_type_str,
        calibration_status=calibration_str,
        claim_ids=list(node_result.affected_claim_ids),
        chunk_ids=list(node_result.affected_chunk_ids),
        doc_ids=list(node_result.affected_doc_ids),
        recommended_for_gating=False,
        limitations=location_limitations,
    )

    evidence_for: list[PinpointEvidence] = [
        PinpointEvidence(
            signal_name=sig.signal_name,
            value=sig.value,
            source_report=sig.source_report or "ncv_pipeline_verifier",
            interpretation=sig.interpretation,
            affected_claim_ids=list(node_result.affected_claim_ids),
            affected_chunk_ids=list(node_result.affected_chunk_ids),
            affected_doc_ids=list(node_result.affected_doc_ids),
            method_type=method_type_str,
            calibration_status=calibration_str,
            limitations=[sig.limitation] if sig.limitation else [],
        )
        for sig in node_result.evidence_signals
    ]

    fallback_heuristics: list[str] = list(report.fallback_heuristics_used)
    if node_result.fallback_used:
        fallback_label = f"ncv_node_{target_node.value}_fallback"
        if fallback_label not in fallback_heuristics:
            fallback_heuristics.append(fallback_label)

    is_uncalibrated = node_result.calibration_status == NCVCalibrationStatus.UNCALIBRATED
    human_review = (
        is_uncalibrated
        or bool(fallback_heuristics)
        or bool(node_result.missing_evidence)
    )

    return [
        PinpointFinding(
            finding_id=f"ncv_finding_{target_node.value}_{report.run_id}",
            location=location,
            evidence_for=evidence_for,
            evidence_against=[],
            missing_evidence=list(node_result.missing_evidence),
            fallback_heuristics_used=fallback_heuristics,
            alternative_locations=[],
            heuristic_score=node_result.node_score,
            calibrated_confidence=None,
            calibration_status=calibration_str,
            human_review_recommended=human_review,
            recommended_for_gating=False,
        )
    ]


def _coerce_to_report(ncv_report: NCVReport | dict) -> NCVReport | None:
    if isinstance(ncv_report, NCVReport):
        return ncv_report
    try:
        return NCVReport.model_validate(ncv_report)
    except Exception:
        return None
