"""Tests for retrieval diagnosis data models."""

from __future__ import annotations

from raggov.models.retrieval_diagnosis import (
    ClaimRetrievalDiagnosisRecord,
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalEvidenceSignal,
    RetrievalFailureType,
)


def test_retrieval_failure_type_values() -> None:
    assert RetrievalFailureType.RETRIEVAL_MISS == "retrieval_miss"
    assert RetrievalFailureType.RETRIEVAL_NOISE == "retrieval_noise"
    assert RetrievalFailureType.RANK_FAILURE_UNKNOWN == "rank_failure_unknown"
    assert RetrievalFailureType.VERSION_RETRIEVAL_FAILURE == "version_retrieval_failure"
    assert RetrievalFailureType.CITATION_RETRIEVAL_MISMATCH == "citation_retrieval_mismatch"
    assert RetrievalFailureType.NO_CLEAR_RETRIEVAL_FAILURE == "no_clear_retrieval_failure"
    assert (
        RetrievalFailureType.INSUFFICIENT_EVIDENCE_TO_DIAGNOSE
        == "insufficient_evidence_to_diagnose"
    )


def test_method_and_calibration_values() -> None:
    assert RetrievalDiagnosisMethodType.HEURISTIC_BASELINE == "heuristic_baseline"
    assert RetrievalDiagnosisMethodType.PRACTICAL_APPROXIMATION == "practical_approximation"
    assert RetrievalDiagnosisCalibrationStatus.UNCALIBRATED == "uncalibrated"
    assert (
        RetrievalDiagnosisCalibrationStatus.PRELIMINARY_CALIBRATED
        == "preliminary_calibrated"
    )
    assert RetrievalDiagnosisCalibrationStatus.CALIBRATED == "calibrated"


def test_report_defaults_and_serialization() -> None:
    signal = RetrievalEvidenceSignal(
        signal_name="no_retrieved_chunks",
        value=True,
        source_report="RAGRun",
        source_ids=[],
        interpretation="No chunks were returned by retrieval.",
    )
    claim = ClaimRetrievalDiagnosisRecord(
        claim_id="claim-1",
        explanation="Required evidence appears missing.",
        evidence_signals=[signal],
    )
    report = RetrievalDiagnosisReport(
        run_id="run-1",
        primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
        affected_claim_ids=["claim-1"],
        claim_records=[claim],
        evidence_signals=[signal],
        recommended_fix="Inspect retriever.",
        method_type=RetrievalDiagnosisMethodType.HEURISTIC_BASELINE,
        calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
    )

    dumped = report.model_dump(mode="json")

    assert dumped["primary_failure_type"] == "retrieval_miss"
    assert dumped["claim_records"][0]["evidence_signals"][0]["signal_name"] == (
        "no_retrieved_chunks"
    )
    assert dumped["recommended_for_gating"] is False
    assert report.noisy_chunk_ids == []
