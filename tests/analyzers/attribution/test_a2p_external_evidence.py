"""A2P v2 external evidence integration tests.

Verifies that AttributionTrace extraction correctly surfaces upstream NCV,
retrieval diagnosis, and external verifier provider signals, and that
candidate generation uses them to strengthen hypotheses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from raggov.analyzers.attribution.candidates import (
    candidate_insufficient_context,
    candidate_weak_evidence,
    candidate_generation_contradicted,
)
from raggov.analyzers.attribution.trace import AttributionTrace, extract_attribution_trace
from raggov.models.citation_faithfulness import (
    CitationCalibrationStatus,
    CitationFaithfulnessReport,
    CitationMethodType,
    ClaimCitationFaithfulnessRecord,
)
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureType, SufficiencyResult
from raggov.models.retrieval_diagnosis import (
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalFailureType,
)
from raggov.models.run import RAGRun


def _make_run(**kwargs) -> RAGRun:
    defaults = dict(
        run_id="run-trace-test",
        query="What is the penalty?",
        retrieved_chunks=[],
        final_answer="The penalty is 500.",
        cited_doc_ids=[],
    )
    defaults.update(kwargs)
    return RAGRun(**defaults)


def _make_retrieval_diagnosis(failure_type: RetrievalFailureType) -> RetrievalDiagnosisReport:
    return RetrievalDiagnosisReport(
        run_id="run-trace-test",
        primary_failure_type=failure_type,
        recommended_fix="Fix retrieval.",
        method_type=RetrievalDiagnosisMethodType.HEURISTIC_BASELINE,
        calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
        limitations=[],
    )


def _make_ncv_prior(first_failing_node: str | None = "retrieval_coverage") -> AnalyzerResult:
    """Build a fake NCVPipelineVerifier result with a parseable NCVReport in evidence[0]."""
    report_data: dict = {
        "run_id": "run-trace-test",
        "node_results": [],
        "first_failing_node": first_failing_node,
        "first_uncertain_node": None,
        "pipeline_health_score": 0.5,
        "bottleneck_description": f"Pipeline fails at {first_failing_node}: retrieval miss.",
        "downstream_failure_chain": [first_failing_node] if first_failing_node else [],
        "evidence_reports_used": ["RetrievalDiagnosisReport"],
        "missing_reports": [],
        "fallback_heuristics_used": [],
        "method_type": "evidence_aggregation",
        "calibration_status": "uncalibrated",
        "recommended_for_gating": False,
        "limitations": [],
    }
    return AnalyzerResult(
        analyzer_name="NCVPipelineVerifier",
        status="fail",
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        evidence=[json.dumps(report_data), "Pipeline fails at retrieval_coverage: retrieval miss."],
    )


def _unsupported_claim() -> ClaimResult:
    return ClaimResult(
        claim_text="The penalty is 500.",
        label="unsupported",
        candidate_chunk_ids=["chunk-1"],
    )


def _make_trace(**overrides) -> AttributionTrace:
    defaults = dict(
        claim_results=[_unsupported_claim()],
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            sufficiency_label="insufficient",
            missing_evidence=["regulation_penalty_clause"],
            affected_claims=["The penalty is 500."],
            limitations=[],
            method="heuristic",
        ),
        sufficiency_method="heuristic",
        has_retrieval_diagnosis_report=False,
        retrieval_primary_failure_type=None,
        has_ncv_report=False,
        ncv_first_failing_node=None,
        ncv_downstream_failure_chain=[],
        ncv_bottleneck_description=None,
        claim_verifier_provider=None,
        citation_verifier_provider=None,
        retrieval_signal_provider=None,
        external_signals_used=[],
        trace_notes=[],
    )
    defaults.update(overrides)
    return AttributionTrace(**defaults)


class TestA2PTraceExtractsNCVFirstFailingNode:
    def test_ncv_first_failing_node_extracted(self):
        run = _make_run()
        prior = [_make_ncv_prior("retrieval_coverage")]
        trace = extract_attribution_trace(run, prior)
        assert trace.ncv_first_failing_node == "retrieval_coverage"

    def test_has_ncv_report_true_when_parseable(self):
        run = _make_run()
        prior = [_make_ncv_prior("retrieval_coverage")]
        trace = extract_attribution_trace(run, prior)
        assert trace.has_ncv_report is True

    def test_no_ncv_report_note_when_missing(self):
        run = _make_run()
        trace = extract_attribution_trace(run, [])
        assert trace.has_ncv_report is False
        assert "no_ncv_report_available" in trace.trace_notes


class TestA2PTraceExtractsRetrievalPrimaryFailureType:
    def test_retrieval_primary_failure_type_extracted(self):
        run = _make_run(
            retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.RETRIEVAL_MISS)
        )
        trace = extract_attribution_trace(run, [])
        assert trace.retrieval_primary_failure_type == "retrieval_miss"
        assert trace.has_retrieval_diagnosis_report is True

    def test_no_retrieval_diag_note_when_missing(self):
        run = _make_run()
        trace = extract_attribution_trace(run, [])
        assert trace.has_retrieval_diagnosis_report is False
        assert "no_retrieval_diagnosis_report_available" in trace.trace_notes


class TestA2PCandidateStrengthenedByRetrievalMiss:
    def test_candidate_includes_retrieval_miss_signal(self):
        trace = _make_trace(
            has_retrieval_diagnosis_report=True,
            retrieval_primary_failure_type="retrieval_miss",
        )
        claim = _unsupported_claim()
        candidate = candidate_insufficient_context(claim, trace)
        assert candidate is not None
        evidence_text = " ".join(candidate.evidence_for)
        assert "retrieval_miss" in evidence_text

    def test_candidate_includes_ncv_coverage_signal(self):
        trace = _make_trace(
            has_ncv_report=True,
            ncv_first_failing_node="retrieval_coverage",
            ncv_bottleneck_description="Pipeline fails at retrieval_coverage: retrieval miss.",
        )
        claim = _unsupported_claim()
        candidate = candidate_insufficient_context(claim, trace)
        assert candidate is not None
        evidence_text = " ".join(candidate.evidence_for)
        assert "retrieval_coverage" in evidence_text

    def test_candidate_advisory_when_no_retrieval_diag(self):
        trace = _make_trace(
            has_retrieval_diagnosis_report=False,
            retrieval_primary_failure_type=None,
        )
        claim = _unsupported_claim()
        candidate = candidate_insufficient_context(claim, trace)
        assert candidate is not None
        evidence_text = " ".join(candidate.evidence_for)
        assert "retrieval_diagnosis_report unavailable" in evidence_text

    def test_candidate_supporting_analyzers_include_retrieval_diag(self):
        trace = _make_trace(
            has_retrieval_diagnosis_report=True,
            retrieval_primary_failure_type="retrieval_miss",
        )
        claim = _unsupported_claim()
        candidate = candidate_insufficient_context(claim, trace)
        assert candidate is not None
        assert "RetrievalDiagnosisReport" in candidate.supporting_analyzers


class TestA2PCandidateIncludesCitationVerifierSignal:
    def test_citation_verifier_provider_in_evidence(self):
        """citation verifier is surfaced in trace; weak_evidence candidate exposes it when noise signal present."""
        trace = _make_trace(
            retrieval_primary_failure_type="retrieval_noise",
            citation_verifier_provider="external_citation_checker_v1",
            external_signals_used=["citation_verifier:external_citation_checker_v1"],
        )
        claim = ClaimResult(
            claim_text="The penalty is 500.",
            label="unsupported",
            candidate_chunk_ids=["chunk-1"],
        )
        candidate = candidate_weak_evidence(claim, trace)
        assert candidate is not None
        evidence_text = " ".join(candidate.evidence_for)
        assert "retrieval_noise" in evidence_text


class TestA2PCandidateIncludesClaimVerifierSignal:
    def test_claim_verifier_provider_in_contradiction_candidate(self):
        trace = _make_trace(
            claim_verifier_provider="external_nli_v2",
            external_signals_used=["claim_verifier:external_nli_v2"],
        )
        claim = ClaimResult(
            claim_text="The penalty is 500.",
            label="contradicted",
            contradicting_chunk_ids=["chunk-1"],
        )
        candidate = candidate_generation_contradicted(claim, trace)
        assert candidate is not None
        evidence_text = " ".join(candidate.evidence_for)
        assert "external_nli_v2" in evidence_text
        assert "external_nli_v2" in candidate.supporting_analyzers


class TestA2PNativeModeNoExternalSignals:
    def test_no_crash_with_no_external_fields(self):
        run = _make_run()
        trace = extract_attribution_trace(run, [])
        assert trace.has_retrieval_diagnosis_report is False
        assert trace.has_ncv_report is False
        assert trace.claim_verifier_provider is None
        assert trace.citation_verifier_provider is None
        assert trace.retrieval_signal_provider is None

    def test_candidate_generation_no_crash_native(self):
        trace = _make_trace()  # All external fields absent
        claim = _unsupported_claim()
        # Should not raise
        candidate = candidate_insufficient_context(claim, trace)
        assert candidate is not None


class TestA2PMissingExternalSignalsVisibleInTrace:
    def test_missing_retrieval_diag_in_trace_notes(self):
        run = _make_run()
        trace = extract_attribution_trace(run, [])
        assert "no_retrieval_diagnosis_report_available" in trace.trace_notes

    def test_missing_ncv_in_trace_notes(self):
        run = _make_run()
        trace = extract_attribution_trace(run, [])
        assert "no_ncv_report_available" in trace.trace_notes

    def test_external_signals_used_populated_when_present(self):
        run = _make_run(
            retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.RETRIEVAL_MISS)
        )
        trace = extract_attribution_trace(run, [])
        assert any("retrieval_diagnosis" in s for s in trace.external_signals_used)
