"""Tests for NCVPriorityPolicy v1.

Tests cover:
1. Policy unit tests (direct NCVReport → NCVPriorityDecision)
2. Verifier integration tests (NCVPipelineVerifier → first_failing_node after policy)
"""

from __future__ import annotations

import json

from raggov.analyzers.verification.ncv import NCVPipelineVerifier
from raggov.analyzers.verification.ncv_priority import (
    NCVPriorityDecision,
    select_first_failing_node_v1,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import (
    CitationCalibrationStatus,
    CitationFaithfulnessReport,
    CitationMethodType,
    CitationSupportLabel,
    ClaimCitationFaithfulnessRecord,
)
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
from raggov.models.retrieval_diagnosis import (
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalFailureType,
)
from raggov.models.run import RAGRun


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _signal(
    name: str,
    value: object = None,
    source: str = "test",
    interp: str = "test signal",
) -> NCVEvidenceSignal:
    return NCVEvidenceSignal(
        signal_name=name,
        value=value,
        source_report=source,
        source_ids=[],
        interpretation=interp,
    )


def _node(
    node: NCVNode,
    status: NCVNodeStatus = NCVNodeStatus.FAIL,
    reason: str = "test reason",
    signals: list[NCVEvidenceSignal] | None = None,
) -> NCVNodeResult:
    return NCVNodeResult(
        node=node,
        status=status,
        primary_reason=reason,
        evidence_signals=signals or [],
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )


def _report(
    first_failing: NCVNode | None,
    node_results: list[NCVNodeResult],
) -> NCVReport:
    return NCVReport(
        run_id="policy-test",
        node_results=node_results,
        first_failing_node=first_failing,
        bottleneck_description=f"Pipeline fails at {first_failing.value if first_failing else 'none'}.",
        method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        calibration_status=NCVCalibrationStatus.UNCALIBRATED,
    )


def _run(**kwargs) -> RAGRun:
    defaults = dict(
        query="What is the penalty?",
        retrieved_chunks=[],
        final_answer="The penalty is 500.",
    )
    defaults.update(kwargs)
    return RAGRun(**defaults)


def _chunk(chunk_id: str, text: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


# ---------------------------------------------------------------------------
# Unit tests — policy function
# ---------------------------------------------------------------------------

class TestCitationBeatsContextAssembly:
    """Rule 4: citation_support beats context_assembly when citation evidence is explicit."""

    def _make_report(self) -> NCVReport:
        ctx_node = _node(
            NCVNode.CONTEXT_ASSEMBLY,
            reason="Legacy Jaccard fallback found duplicate chunks.",
            signals=[_signal("duplicate_chunk_jaccard", 0.91, "heuristic_fallback", "Duplicate chunks found.")],
        )
        cit_node = _node(
            NCVNode.CITATION_SUPPORT,
            reason="Citation faithfulness report found unsupported, phantom, missing, or mismatched citation support.",
            signals=[_signal("citation_faithfulness_issues", 2, "CitationFaithfulnessReport", "Citation issues found.")],
        )
        return _report(NCVNode.CONTEXT_ASSEMBLY, [ctx_node, cit_node])

    def test_selected_node_is_citation_support(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.selected_node == NCVNode.CITATION_SUPPORT.value

    def test_changed_is_true(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.changed is True

    def test_original_is_context_assembly(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.original_first_failing_node == NCVNode.CONTEXT_ASSEMBLY.value

    def test_recommended_for_gating_is_false(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.recommended_for_gating is False

    def test_calibration_status_is_uncalibrated(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.calibration_status == "uncalibrated"

    def test_accepts_dict_input(self):
        report_dict = self._make_report().model_dump(mode="json")
        decision = select_first_failing_node_v1(report_dict)
        assert decision.selected_node == NCVNode.CITATION_SUPPORT.value
        assert decision.changed is True


class TestRetrievalCoverageBeatsRetrievalPrecision:
    """Rule 5: retrieval_coverage beats retrieval_precision for retrieval_miss."""

    def _make_report(self) -> NCVReport:
        cov_node = _node(
            NCVNode.RETRIEVAL_COVERAGE,
            reason="Retrieval diagnosis reports a retrieval miss.",
            signals=[_signal("retrieval_primary_failure", "retrieval_miss", "RetrievalDiagnosisReport", "Retrieval miss.")],
        )
        prec_node = _node(
            NCVNode.RETRIEVAL_PRECISION,
            reason="Mean retrieval score 0.29 is below 0.50.",
            signals=[_signal("mean_retrieval_score", 0.29, "heuristic_fallback", "Low mean score.")],
        )
        return _report(NCVNode.RETRIEVAL_PRECISION, [cov_node, prec_node])

    def test_selected_node_is_retrieval_coverage(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.selected_node == NCVNode.RETRIEVAL_COVERAGE.value

    def test_changed_is_true(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.changed is True

    def test_original_is_retrieval_precision(self):
        decision = select_first_failing_node_v1(self._make_report())
        assert decision.original_first_failing_node == NCVNode.RETRIEVAL_PRECISION.value


class TestRetrievalPrecisionRemainsForNoise:
    """Rule 6: retrieval_precision stays when retrieval_coverage is not failing."""

    def test_stays_with_retrieval_noise(self):
        prec_node = _node(
            NCVNode.RETRIEVAL_PRECISION,
            reason="Retrieval diagnosis reports retrieval noise.",
            signals=[_signal("retrieval_primary_failure", "retrieval_noise", "RetrievalDiagnosisReport")],
        )
        report = _report(NCVNode.RETRIEVAL_PRECISION, [prec_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.RETRIEVAL_PRECISION.value
        assert decision.changed is False

    def test_stays_with_low_score_heuristic_no_coverage_fail(self):
        prec_node = _node(
            NCVNode.RETRIEVAL_PRECISION,
            reason="Mean retrieval score 0.29 is below 0.50.",
            signals=[_signal("mean_retrieval_score", 0.29, "heuristic_fallback")],
        )
        report = _report(NCVNode.RETRIEVAL_PRECISION, [prec_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.RETRIEVAL_PRECISION.value
        assert decision.changed is False


class TestVersionValidityBeatsCitationSupport:
    """Rule 3: version_validity beats citation_support for stale cited source."""

    def test_selected_node_is_version_validity(self):
        cit_node = _node(
            NCVNode.CITATION_SUPPORT,
            reason="Citation faithfulness report found unsupported citation support.",
            signals=[_signal("citation_faithfulness_issues", 1, "CitationFaithfulnessReport")],
        )
        ver_node = _node(
            NCVNode.VERSION_VALIDITY,
            reason="Version validity report found invalid cited documents.",
            signals=[_signal("invalid_cited_doc_ids", 2, "VersionValidityReport", "Cited docs are superseded.")],
        )
        report = _report(NCVNode.CITATION_SUPPORT, [cit_node, ver_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.VERSION_VALIDITY.value

    def test_changed_is_true_when_original_was_citation(self):
        cit_node = _node(NCVNode.CITATION_SUPPORT, reason="Citation faithfulness issues found.")
        ver_node = _node(
            NCVNode.VERSION_VALIDITY,
            reason="Version validity report found invalid cited documents.",
            signals=[_signal("invalid_cited_doc_ids", 1, "VersionValidityReport")],
        )
        report = _report(NCVNode.CITATION_SUPPORT, [cit_node, ver_node])
        decision = select_first_failing_node_v1(report)
        assert decision.changed is True

    def test_stale_reason_keyword_triggers_rule(self):
        ver_node = _node(
            NCVNode.VERSION_VALIDITY,
            reason="Version validity found superseded cited documents.",
        )
        report = _report(NCVNode.VERSION_VALIDITY, [ver_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.VERSION_VALIDITY.value


class TestSecurityRiskSelected:
    """Rule 2: security_risk selected when malicious context is detected."""

    def test_security_risk_overrides_claim_support(self):
        claim_node = _node(
            NCVNode.CLAIM_SUPPORT,
            reason="Claim grounding reports unsupported claims.",
        )
        sec_node = _node(
            NCVNode.SECURITY_RISK,
            reason="Security analyzer reported prompt injection.",
            signals=[_signal("security_result_status", "fail", "PromptInjectionAnalyzer")],
        )
        report = _report(NCVNode.CLAIM_SUPPORT, [claim_node, sec_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.SECURITY_RISK.value

    def test_changed_is_true(self):
        sec_node = _node(NCVNode.SECURITY_RISK, reason="Security analyzer reported poisoning.")
        report = _report(NCVNode.RETRIEVAL_COVERAGE, [
            _node(NCVNode.RETRIEVAL_COVERAGE, reason="Retrieval miss."),
            sec_node,
        ])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.SECURITY_RISK.value
        assert decision.changed is True

    def test_recommended_for_gating_remains_false(self):
        sec_node = _node(NCVNode.SECURITY_RISK, reason="Security analyzer reported prompt injection.")
        report = _report(NCVNode.SECURITY_RISK, [sec_node])
        decision = select_first_failing_node_v1(report)
        assert decision.recommended_for_gating is False


class TestParserValiditySelected:
    """Rule 1: parser_validity selected for blocking parser damage."""

    def test_selected_for_blocking_damage(self):
        parser_node = _node(
            NCVNode.PARSER_VALIDITY,
            reason="Parser validation found blocking structural/provenance damage.",
            signals=[_signal("parser_validation_finding", "TABLE_STRUCTURE_LOSS", "ParserValidationResults")],
        )
        report = _report(NCVNode.PARSER_VALIDITY, [parser_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.PARSER_VALIDITY.value

    def test_parser_beats_security(self):
        parser_node = _node(
            NCVNode.PARSER_VALIDITY,
            reason="Parser validation found blocking structural/provenance damage.",
        )
        sec_node = _node(NCVNode.SECURITY_RISK, reason="Security issue.")
        report = _report(NCVNode.SECURITY_RISK, [parser_node, sec_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.PARSER_VALIDITY.value

    def test_changed_is_true_when_original_is_different(self):
        parser_node = _node(
            NCVNode.PARSER_VALIDITY,
            reason="Parser validation found blocking structural/provenance damage.",
        )
        ret_node = _node(NCVNode.RETRIEVAL_COVERAGE, reason="Retrieval miss.")
        report = _report(NCVNode.RETRIEVAL_COVERAGE, [parser_node, ret_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.PARSER_VALIDITY.value
        assert decision.changed is True


class TestClaimSupportWithoutRetrievalCoverageFail:
    """Rule 8: claim_support selected when retrieval_coverage is not failing."""

    def test_claim_support_stays_when_coverage_passes(self):
        cov_node = _node(NCVNode.RETRIEVAL_COVERAGE, status=NCVNodeStatus.PASS, reason="Coverage OK.")
        claim_node = _node(
            NCVNode.CLAIM_SUPPORT,
            reason="Claim grounding reports unsupported claims.",
        )
        report = _report(NCVNode.CLAIM_SUPPORT, [cov_node, claim_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node == NCVNode.CLAIM_SUPPORT.value
        assert decision.changed is False


class TestNoFailureNode:
    """No failure: selected_node is None."""

    def test_returns_none_when_no_fail_nodes(self):
        cov_node = _node(NCVNode.RETRIEVAL_COVERAGE, status=NCVNodeStatus.PASS, reason="OK.")
        report = _report(None, [cov_node])
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node is None
        assert decision.changed is False

    def test_empty_node_results(self):
        report = NCVReport(
            run_id="clean",
            node_results=[],
            first_failing_node=None,
            bottleneck_description="No failures.",
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            calibration_status=NCVCalibrationStatus.UNCALIBRATED,
        )
        decision = select_first_failing_node_v1(report)
        assert decision.selected_node is None
        assert decision.changed is False


# ---------------------------------------------------------------------------
# Integration tests — NCVPipelineVerifier + policy
# ---------------------------------------------------------------------------

class TestVerifierAppliesPolicyCitationBeatsContext:
    """NCVPipelineVerifier emits citation_support as first_failing_node when
    citation evidence is explicit and context_assembly also fails."""

    def test_citation_support_selected_with_citation_report(self):
        from raggov.models.chunk import RetrievedChunk

        # Two near-identical chunks trigger context_assembly via Jaccard
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice before action.", source_doc_id="d1", score=0.85),
            RetrievedChunk(chunk_id="c2", text="Rule 5 requires notice before action.", source_doc_id="d2", score=0.84),
        ]

        # Build a CitationFaithfulnessReport showing citation mismatch
        record = ClaimCitationFaithfulnessRecord(
            claim_id="claim-1",
            claim_text="Rule 5 requires notice.",
            citation_support_label=CitationSupportLabel.UNSUPPORTED,
        )
        cit_report = CitationFaithfulnessReport(
            run_id="run-cit",
            records=[record],
            unsupported_claim_ids=["claim-1"],
            method_type=CitationMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=CitationCalibrationStatus.UNCALIBRATED,
        )

        run = RAGRun(
            query="Explain Rule 5",
            retrieved_chunks=chunks,
            final_answer="Rule 5 requires notice.",
            citation_faithfulness_report=cit_report,
        )

        verifier = NCVPipelineVerifier({"fail_fast": False})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        # Policy should select citation_support over context_assembly
        assert report["first_failing_node"] == NCVNode.CITATION_SUPPORT.value

    def test_context_assembly_stays_without_citation_report(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice before action.", source_doc_id="d1", score=0.85),
            RetrievedChunk(chunk_id="c2", text="Rule 5 requires notice before action.", source_doc_id="d2", score=0.84),
        ]
        run = RAGRun(
            query="Explain Rule 5",
            retrieved_chunks=chunks,
            final_answer="Rule 5 requires notice.",
        )
        verifier = NCVPipelineVerifier({"fail_fast": False})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        assert report["first_failing_node"] == NCVNode.CONTEXT_ASSEMBLY.value


class TestVerifierAppliesPolicyCoverageBeatsPresicion:
    """NCVPipelineVerifier emits retrieval_coverage as first_failing_node when
    retrieval_miss is explicit and retrieval_precision also fails."""

    def test_retrieval_coverage_selected_when_miss_and_precision_both_fail(self):
        diagnosis = RetrievalDiagnosisReport(
            run_id="run-miss",
            primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
            affected_claim_ids=["claim-1"],
            recommended_fix="Expand retrieval.",
            method_type=RetrievalDiagnosisMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
        )
        prior = AnalyzerResult(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            retrieval_diagnosis_report=diagnosis,
        )
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice.", source_doc_id="d1", score=0.29),
        ]
        run = RAGRun(
            query="What is Rule 5?",
            retrieved_chunks=chunks,
            final_answer="Rule 5 requires notice.",
        )

        verifier = NCVPipelineVerifier({
            "prior_results": [prior],
            "fail_fast": False,
            "allow_retrieval_precision_fallback": True,
        })
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        assert report["first_failing_node"] == NCVNode.RETRIEVAL_COVERAGE.value
        assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
        assert result.stage == FailureStage.RETRIEVAL


class TestVerifierPolicyDecisionPreservedInReport:
    """When policy changes first_failing_node, original_first_failing_node is stored."""

    def test_original_node_preserved_when_changed(self):
        cit_report = CitationFaithfulnessReport(
            run_id="run-pres",
            records=[
                ClaimCitationFaithfulnessRecord(
                    claim_id="c1",
                    claim_text="Claim.",
                    citation_support_label=CitationSupportLabel.UNSUPPORTED,
                )
            ],
            unsupported_claim_ids=["c1"],
            method_type=CitationMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=CitationCalibrationStatus.UNCALIBRATED,
        )
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice before action.", source_doc_id="d1", score=0.85),
            RetrievedChunk(chunk_id="c2", text="Rule 5 requires notice before action.", source_doc_id="d2", score=0.84),
        ]
        run = RAGRun(
            query="Explain Rule 5",
            retrieved_chunks=chunks,
            final_answer="Rule 5 requires notice.",
            citation_faithfulness_report=cit_report,
        )

        verifier = NCVPipelineVerifier({"fail_fast": False})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        # Policy should have changed the node
        assert report["first_failing_node"] == NCVNode.CITATION_SUPPORT.value
        # Original node is recorded
        assert report["original_first_failing_node"] == NCVNode.CONTEXT_ASSEMBLY.value
        # Policy decision metadata is present
        assert report["priority_policy_decision"] is not None
        assert report["priority_policy_decision"]["changed"] is True

    def test_original_node_none_when_not_changed(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice.", source_doc_id="d1", score=0.88),
        ]
        diagnosis = RetrievalDiagnosisReport(
            run_id="run-nc",
            primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
            recommended_fix="Fix retrieval.",
            method_type=RetrievalDiagnosisMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
        )
        prior = AnalyzerResult(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            retrieval_diagnosis_report=diagnosis,
        )
        run = RAGRun(
            query="What is Rule 5?",
            retrieved_chunks=chunks,
            final_answer="Rule 5 requires notice.",
        )
        verifier = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": True})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        # Policy should not have changed it
        assert report["first_failing_node"] == NCVNode.RETRIEVAL_COVERAGE.value
        assert report["original_first_failing_node"] is None
        assert report["priority_policy_decision"] is None


class TestVerifierExistingBehaviorPreserved:
    """Policy must not change results for existing test scenarios."""

    def test_retrieval_precision_unchanged_no_coverage_fail(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice.", source_doc_id="d1", score=0.31),
            RetrievedChunk(chunk_id="c2", text="Rule 5 also references appeals.", source_doc_id="d2", score=0.28),
        ]
        run = RAGRun(query="What is Rule 5?", retrieved_chunks=chunks, final_answer="Rule 5 requires notice.")
        verifier = NCVPipelineVerifier({"allow_retrieval_precision_fallback": True})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        assert report["first_failing_node"] == NCVNode.RETRIEVAL_PRECISION.value
        assert result.failure_type == FailureType.RETRIEVAL_ANOMALY

    def test_empty_chunks_keeps_query_understanding(self):
        run = RAGRun(query="What is Rule 5?", retrieved_chunks=[], final_answer="No answer.")
        verifier = NCVPipelineVerifier()
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        assert report["first_failing_node"] == NCVNode.QUERY_UNDERSTANDING.value
        assert result.failure_type == FailureType.SCOPE_VIOLATION

    def test_claim_support_unchanged_no_retrieval_coverage_fail(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", text="Rule 5 requires notice before action.", source_doc_id="d1", score=0.88),
        ]
        prior = AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            claim_results=[
                ClaimResult(claim_text="Rule 5 creates a pension benefit.", label="unsupported"),
            ],
        )
        run = RAGRun(
            query="What is Rule 5?",
            retrieved_chunks=chunks,
            final_answer="Rule 5 creates a pension benefit.",
        )
        verifier = NCVPipelineVerifier({"prior_results": [prior], "fail_fast": False})
        result = verifier.analyze(run)

        report = json.loads(result.evidence[0])
        assert report["first_failing_node"] == NCVNode.CLAIM_SUPPORT.value
        assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
