"""NCV external evidence integration tests.

Verifies that NCVPipelineVerifier consumes upstream structured reports and
surfaces external provider signals without silently ignoring missing data.
"""

from __future__ import annotations

import pytest

from raggov.analyzers.verification.ncv import NCVPipelineVerifier
from raggov.models.citation_faithfulness import (
    CitationCalibrationStatus,
    CitationFaithfulnessReport,
    CitationMethodType,
    ClaimCitationFaithfulnessRecord,
    CitationSupportLabel,
)
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.grounding import CalibrationStatus, ClaimEvidenceRecord, ClaimVerificationLabel, GroundingEvidenceBundle
from raggov.models.ncv import NCVNode, NCVNodeStatus
from raggov.models.retrieval_diagnosis import (
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalFailureType,
)
from raggov.models.run import RAGRun


def _make_run(**kwargs) -> RAGRun:
    defaults = dict(
        run_id="run-test",
        query="What is the penalty?",
        retrieved_chunks=[],
        final_answer="The penalty is 500.",
        cited_doc_ids=[],
    )
    defaults.update(kwargs)
    return RAGRun(**defaults)


def _make_retrieval_diagnosis(failure_type: RetrievalFailureType) -> RetrievalDiagnosisReport:
    return RetrievalDiagnosisReport(
        run_id="run-test",
        primary_failure_type=failure_type,
        recommended_fix="Fix retrieval.",
        method_type=RetrievalDiagnosisMethodType.HEURISTIC_BASELINE,
        calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
        limitations=["uncalibrated"],
    )


def _make_ncv(run: RAGRun, prior_results: list[AnalyzerResult] | None = None) -> AnalyzerResult:
    verifier = NCVPipelineVerifier(config={"prior_results": prior_results or [], "fail_fast": False})
    return verifier.analyze(run)


class TestRetrievalCoverageUsesRetrievalDiagnosisReport:
    """retrieval_coverage node prefers RetrievalDiagnosisReport.primary_failure_type=retrieval_miss."""

    def test_retrieval_miss_causes_fail(self):
        run = _make_run(retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.RETRIEVAL_MISS))
        result = _make_ncv(run)
        # NCV fails — QUERY_UNDERSTANDING fires first with SCOPE_VIOLATION when no chunks,
        # but with chunks the retrieval_coverage node would fail. We verify failure status.
        assert result.status == "fail"

    def test_retrieval_miss_causes_fail_with_chunks(self):
        import json
        from raggov.models.chunk import RetrievedChunk
        chunk = RetrievedChunk(chunk_id="c1", text="Some text", source_doc_id="doc1", score=None)
        run = _make_run(
            retrieved_chunks=[chunk],
            retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.RETRIEVAL_MISS),
        )
        result = _make_ncv(run)
        assert result.status == "fail"
        # Verify retrieval_coverage node itself is FAIL due to retrieval_miss
        ncv_report = json.loads(result.evidence[0])
        coverage_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "retrieval_coverage"
        )
        assert coverage_node["status"] == "fail"

    def test_no_retrieval_failure_not_forced_to_fail(self):
        from raggov.models.chunk import RetrievedChunk
        chunk = RetrievedChunk(chunk_id="c1", text="Some text about penalties", source_doc_id="doc1", score=None)
        run = _make_run(
            retrieved_chunks=[chunk],
            retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.NO_CLEAR_RETRIEVAL_FAILURE),
        )
        result = _make_ncv(run)
        import json
        ncv_report = json.loads(result.evidence[0])
        coverage_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "retrieval_coverage"
        )
        assert coverage_node["status"] == "pass"

    def test_evidence_reports_used_includes_report(self):
        import json
        run = _make_run(retrieval_diagnosis_report=_make_retrieval_diagnosis(RetrievalFailureType.RETRIEVAL_MISS))
        result = _make_ncv(run)
        ncv_report = json.loads(result.evidence[0])
        assert "RetrievalDiagnosisReport" in ncv_report["evidence_reports_used"]


class TestClaimSupportRecordsExternalClaimVerifier:
    """NCV claim_support node exposes external_claim_verifier_provider signal."""

    def _make_grounding_with_provider(self, provider: str) -> AnalyzerResult:
        record = ClaimEvidenceRecord(
            claim_id="c1",
            claim_text="The penalty is 500.",
            verification_label=ClaimVerificationLabel.ENTAILED,
            verifier_method=provider,
            verifier_score=0.9,
            calibration_status=CalibrationStatus.UNCALIBRATED,
        )
        bundle = GroundingEvidenceBundle(claim_evidence_records=[record])
        claim_result = ClaimResult(claim_text="The penalty is 500.", label="entailed")
        return AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[claim_result],
            grounding_evidence_bundle=bundle,
        )

    def test_external_verifier_signal_is_present_on_pass(self):
        import json
        run = _make_run()
        prior = [self._make_grounding_with_provider("external_nli_verifier_v1")]
        result = _make_ncv(run, prior)
        ncv_report = json.loads(result.evidence[0])
        claim_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "claim_support"
        )
        signal_names = [s["signal_name"] for s in claim_node["evidence_signals"]]
        assert "external_claim_verifier_provider" in signal_names

    def test_external_verifier_signal_value_matches_provider(self):
        import json
        run = _make_run()
        prior = [self._make_grounding_with_provider("my_llm_judge")]
        result = _make_ncv(run, prior)
        ncv_report = json.loads(result.evidence[0])
        claim_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "claim_support"
        )
        provider_signal = next(
            s for s in claim_node["evidence_signals"] if s["signal_name"] == "external_claim_verifier_provider"
        )
        assert provider_signal["value"] == "my_llm_judge"

    def test_no_external_verifier_signal_when_heuristic(self):
        import json
        run = _make_run()
        claim_result = ClaimResult(claim_text="The penalty is 500.", label="entailed")
        prior = [AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            claim_results=[claim_result],
        )]
        result = _make_ncv(run, prior)
        ncv_report = json.loads(result.evidence[0])
        claim_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "claim_support"
        )
        signal_names = [s["signal_name"] for s in claim_node["evidence_signals"]]
        assert "external_claim_verifier_provider" not in signal_names


class TestCitationSupportRecordsExternalCitationVerifier:
    """NCV citation_support node exposes external_citation_verifier_provider signal."""

    def _make_citation_report_with_provider(self, provider: str) -> CitationFaithfulnessReport:
        record = ClaimCitationFaithfulnessRecord(
            claim_id="c1",
            claim_text="The penalty is 500.",
            citation_support_label=CitationSupportLabel.FULLY_SUPPORTED,
            external_signal_provider=provider,
        )
        return CitationFaithfulnessReport(
            run_id="run-test",
            records=[record],
            method_type=CitationMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=CitationCalibrationStatus.UNCALIBRATED,
        )

    def test_citation_verifier_signal_present(self):
        import json
        run = _make_run(citation_faithfulness_report=self._make_citation_report_with_provider("citation_checker_v2"))
        result = _make_ncv(run)
        ncv_report = json.loads(result.evidence[0])
        cit_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "citation_support"
        )
        signal_names = [s["signal_name"] for s in cit_node["evidence_signals"]]
        assert "external_citation_verifier_provider" in signal_names

    def test_no_citation_signal_when_no_provider(self):
        import json
        record = ClaimCitationFaithfulnessRecord(
            claim_id="c1",
            claim_text="The penalty is 500.",
            citation_support_label=CitationSupportLabel.FULLY_SUPPORTED,
        )
        report = CitationFaithfulnessReport(
            run_id="run-test",
            records=[record],
            method_type=CitationMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=CitationCalibrationStatus.UNCALIBRATED,
        )
        run = _make_run(citation_faithfulness_report=report)
        result = _make_ncv(run)
        ncv_report = json.loads(result.evidence[0])
        cit_node = next(
            n for n in ncv_report["node_results"] if n["node"] == "citation_support"
        )
        signal_names = [s["signal_name"] for s in cit_node["evidence_signals"]]
        assert "external_citation_verifier_provider" not in signal_names


class TestNativeModeNoCrash:
    """Native mode runs NCV with no external reports without errors."""

    def test_native_mode_no_external(self):
        run = _make_run()
        result = _make_ncv(run, [])
        # Should not raise; status may be pass/warn/fail but not error
        assert result.analyzer_name == "NCVPipelineVerifier"
        assert result.status in {"pass", "warn", "fail"}


class TestMissingRetrievalDiagnosisReportIsVisible:
    """Missing retrieval diagnosis report surfaces in NCVReport.missing_reports."""

    def test_missing_report_is_listed(self):
        import json
        run = _make_run()  # No retrieval_diagnosis_report set
        result = _make_ncv(run, [])
        ncv_report = json.loads(result.evidence[0])
        assert "retrieval_diagnosis_report" in ncv_report.get("missing_reports", [])
