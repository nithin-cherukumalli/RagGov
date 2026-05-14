from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, Diagnosis, FailureStage, FailureType
from raggov.models.ncv import (
    NCVCalibrationStatus,
    NCVMethodType,
    NCVReport,
    NCVNode,
    NCVNodeResult,
    NCVNodeStatus,
)
from raggov.models.pinpoint import CausalChain, PinpointFinding, PinpointLocation, TrustDecision
from raggov.models.run import RAGRun


class StaticAnalyzer(BaseAnalyzer):
    def __init__(self, result: AnalyzerResult) -> None:
        super().__init__()
        self.result = result

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


def _run() -> RAGRun:
    return RAGRun(
        run_id="structured-run",
        query="What is the refund policy?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Refund policy covers hardware returns for thirty days.",
                source_doc_id="doc-1",
                score=0.92,
            )
        ],
        final_answer="The refund policy covers hardware returns for thirty days.",
    )


def _ncv_report(first_failing_node: NCVNode = NCVNode.RETRIEVAL_COVERAGE) -> dict:
    return NCVReport(
        run_id="structured-run",
        node_results=[
            NCVNodeResult(
                node=first_failing_node,
                status=NCVNodeStatus.FAIL,
                primary_reason="Needed evidence was not retrieved.",
                affected_claim_ids=["claim-1"],
                affected_chunk_ids=["chunk-1"],
                affected_doc_ids=["doc-1"],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                calibration_status=NCVCalibrationStatus.UNCALIBRATED,
                limitations=["Heuristic NCV output."],
            )
        ],
        first_failing_node=first_failing_node,
        bottleneck_description="Retrieval coverage is the first bottleneck.",
        downstream_failure_chain=[first_failing_node],
        evidence_reports_used=["RetrievalDiagnosisReport"],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        recommended_for_gating=False,
        limitations=["Uncalibrated NCV report."],
    ).model_dump(mode="json")


def _pinpoint_finding() -> PinpointFinding:
    return PinpointFinding(
        finding_id="finding-1",
        location=PinpointLocation(
            location_id="loc-1",
            ncv_node="retrieval_coverage",
            pipeline_stage="retrieval",
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
        missing_evidence=["missing-proof"],
        fallback_heuristics_used=["heuristic_fallback"],
        heuristic_score=0.5,
        calibration_status="uncalibrated",
        recommended_for_gating=False,
    )


def _causal_chain() -> CausalChain:
    finding = _pinpoint_finding()
    return CausalChain(
        chain_id="chain-1",
        root_location=finding.location,
        downstream_locations=[],
        causal_hypothesis="retrieval_coverage_gap",
        abduct="Needed evidence was not retrieved.",
        act="Increase retrieval recall.",
        predict="Unsupported claims should reduce when missing evidence is retrieved.",
        evidence_for=[],
        evidence_against=[],
        alternative_explanations=["corpus coverage gap"],
        heuristic_score=0.5,
        calibrated_confidence=None,
        calibration_status="uncalibrated",
    )


def test_diagnosis_model_accepts_structured_pinpoint_fields() -> None:
    diagnosis = Diagnosis(
        run_id="run-1",
        primary_failure=FailureType.UNSUPPORTED_CLAIM,
        root_cause_stage=FailureStage.GROUNDING,
        should_have_answered=True,
        security_risk="NONE",
        recommended_fix="Improve retrieval recall.",
        pinpoint_findings=[_pinpoint_finding()],
        causal_chains=[_causal_chain()],
    )

    roundtrip = Diagnosis.model_validate(diagnosis.model_dump(mode="json"))

    assert roundtrip.pinpoint_findings[0].location.ncv_node == "retrieval_coverage"
    assert roundtrip.causal_chains[0].causal_hypothesis == "retrieval_coverage_gap"


def test_engine_attaches_pinpoint_findings_when_ncv_report_exists() -> None:
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="ClaimGroundingAnalyzer",
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="NCVPipelineVerifier",
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.RETRIEVAL,
                ncv_report=_ncv_report(),
            )
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(_run())

    assert diagnosis.pinpoint_findings
    assert diagnosis.pinpoint_findings[0].location.ncv_node == "retrieval_coverage"
    assert diagnosis.pinpoint_findings[0].recommended_for_gating is False


def test_engine_attaches_causal_chains_when_a2p_result_exposes_them() -> None:
    pinpoint = _pinpoint_finding()
    chain = _causal_chain()
    trust = TrustDecision(
        decision="human_review",
        reason="Uncalibrated structured outputs remain advisory.",
        recommended_for_gating=False,
        human_review_required=True,
        blocking_eligible=False,
        unmet_requirements=["calibration_unavailable", "human_review_required"],
        calibration_status="uncalibrated",
        fallback_heuristics_used=["heuristic_fallback"],
    )
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="ClaimGroundingAnalyzer",
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="NCVPipelineVerifier",
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.RETRIEVAL,
                ncv_report=_ncv_report(),
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="A2PAttributionAnalyzer",
                status="fail",
                failure_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
                stage=FailureStage.RETRIEVAL,
                pinpoint_findings=[pinpoint],
                causal_chains=[chain],
                trust_decision=trust,
                evidence=['a2p_causal_chain:{"root_node":"retrieval_coverage"}'],
            )
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(_run())

    assert diagnosis.causal_chains
    assert diagnosis.causal_chains[0].root_location.ncv_node == "retrieval_coverage"
    assert diagnosis.causal_chains[0].causal_hypothesis == "retrieval_coverage_gap"
    assert diagnosis.causal_chains[0].calibration_status == "uncalibrated"
    assert diagnosis.causal_chains[0].calibrated_confidence is None
    assert diagnosis.trust_decision is not None
    assert diagnosis.trust_decision.recommended_for_gating is False


def test_engine_remains_backward_compatible_without_ncv_or_a2p_outputs() -> None:
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="ClaimGroundingAnalyzer",
                status="pass",
            )
        )
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers, config={"mode": "native"}).diagnose(_run())

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert diagnosis.pinpoint_findings == []
    assert diagnosis.causal_chains == []
    assert diagnosis.trust_decision is None


def test_trust_decision_is_conservative_if_present() -> None:
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="ClaimGroundingAnalyzer",
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="NCVPipelineVerifier",
                status="fail",
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                stage=FailureStage.RETRIEVAL,
                ncv_report=_ncv_report(),
            )
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(_run())

    assert diagnosis.trust_decision is not None
    assert diagnosis.trust_decision.recommended_for_gating is False
    assert diagnosis.trust_decision.blocking_eligible is False
    assert diagnosis.trust_decision.calibration_status == "uncalibrated"
    assert diagnosis.trust_decision.decision in {"warn", "human_review"}
