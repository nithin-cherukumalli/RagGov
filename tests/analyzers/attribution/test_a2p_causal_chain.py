from __future__ import annotations

import json

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.attribution.causal_chain import (
    build_causal_chains_from_a2p,
    summarize_causal_chains_for_a2p,
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


def _finding(node: str) -> PinpointFinding:
    return PinpointFinding(
        finding_id=f"finding-{node}",
        location=PinpointLocation(
            location_id=f"loc-{node}",
            ncv_node=node,
            pipeline_stage="retrieval" if "retrieval" in node or node == "version_validity" else "grounding",
            failure_type="INSUFFICIENT_CONTEXT",
            claim_ids=["claim-1"],
            chunk_ids=["chunk-1"],
            doc_ids=["doc-1"],
            localization_method="ncv_first_failing_node_v1",
            method_type="evidence_aggregation",
            calibration_status="uncalibrated",
            recommended_for_gating=False,
            limitations=["Heuristic pinpoint."],
        ),
        evidence_for=[
            PinpointEvidence(
                signal_name="test_signal",
                value=1,
                source_report="NCVReport",
                interpretation="Test evidence.",
                affected_claim_ids=["claim-1"],
                affected_chunk_ids=["chunk-1"],
                affected_doc_ids=["doc-1"],
                method_type="evidence_aggregation",
                calibration_status="uncalibrated",
                limitations=[],
            )
        ],
        missing_evidence=["missing-proof"],
        fallback_heuristics_used=["heuristic_fallback"],
        heuristic_score=0.5,
        calibration_status="uncalibrated",
        recommended_for_gating=False,
    )


def _ncv_report(first_failing_node: NCVNode) -> NCVReport:
    node_result = NCVNodeResult(
        node=first_failing_node,
        status=NCVNodeStatus.FAIL,
        primary_reason="Primary reason.",
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
        limitations=["Heuristic NCV output."],
    )
    return NCVReport(
        run_id="run-causal",
        node_results=[node_result],
        first_failing_node=first_failing_node,
        bottleneck_description=f"Pipeline fails at {first_failing_node.value}.",
        downstream_failure_chain=[first_failing_node],
        evidence_reports_used=["RetrievalDiagnosisReport"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        recommended_for_gating=False,
        limitations=["Uncalibrated NCV report."],
    )


def _run_with_ncv(report: NCVReport | dict | None) -> RAGRun:
    metadata = {"reports": {}}
    if report is not None:
        metadata["reports"]["ncv_report"] = (
            report if isinstance(report, dict) else report.model_dump(mode="json")
        )
    return RAGRun(
        run_id="run-causal",
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


def test_build_causal_chain_from_retrieval_coverage_pinpoint() -> None:
    chains = build_causal_chains_from_a2p([_finding("retrieval_coverage")])

    assert len(chains) == 1
    chain = chains[0]
    assert chain.root_location.ncv_node == "retrieval_coverage"
    assert "retrieval_coverage_gap" in chain.causal_hypothesis
    assert chain.abduct
    assert chain.act
    assert chain.predict
    assert chain.calibrated_confidence is None
    assert chain.calibration_status == "uncalibrated"


def test_citation_support_maps_to_citation_support_failure() -> None:
    chain = build_causal_chains_from_a2p([_finding("citation_support")])[0]

    assert "citation_support_failure" in chain.causal_hypothesis
    assert "citation" in chain.act.lower()
    assert summarize_causal_chains_for_a2p([chain])["recommended_for_gating"] is False


def test_version_validity_maps_to_stale_or_invalid_source_usage() -> None:
    chain = build_causal_chains_from_a2p([_finding("version_validity")])[0]

    assert "stale_or_invalid_source_usage" in chain.causal_hypothesis
    assert "effective-date" in chain.act or "supersession" in chain.act


def test_security_risk_maps_to_adversarial_or_unsafe_context() -> None:
    chain = build_causal_chains_from_a2p([_finding("security_risk")])[0]

    assert "adversarial_or_unsafe_context" in chain.causal_hypothesis
    assert "quarantine" in chain.act.lower() or "security screening" in chain.act.lower()


def test_a2p_output_includes_causal_chain_when_pinpoint_context_exists() -> None:
    run = _run_with_ncv(_ncv_report(NCVNode.RETRIEVAL_COVERAGE))
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
    line = next(item for item in result.evidence if item.startswith("a2p_causal_chain:"))
    payload = json.loads(line.split(":", 1)[1])
    assert payload["root_node"] == "retrieval_coverage"
    assert payload["calibration_status"] == "uncalibrated"
    assert payload["recommended_for_gating"] is False


def test_a2p_without_pinpoint_context_behaves_unchanged() -> None:
    run = _run_with_ncv(None)
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
    assert not any(item.startswith("a2p_causal_chain:") for item in result.evidence)


def test_malformed_pinpoint_context_does_not_crash() -> None:
    run = _run_with_ncv({"malformed": True})
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
    assert not any(item.startswith("a2p_causal_chain:") for item in result.evidence)
