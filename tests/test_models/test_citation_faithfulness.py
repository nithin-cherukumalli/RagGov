"""Tests for citation faithfulness data models."""

from __future__ import annotations

from raggov.models import (
    CitationCalibrationStatus,
    CitationEvidenceSource,
    CitationFaithfulnessReport,
    CitationFaithfulnessRisk,
    CitationMethodType,
    CitationSupportLabel,
    ClaimCitationFaithfulnessRecord,
)


def test_citation_support_label_values() -> None:
    assert CitationSupportLabel.FULLY_SUPPORTED == "fully_supported"
    assert CitationSupportLabel.PARTIALLY_SUPPORTED == "partially_supported"
    assert CitationSupportLabel.UNSUPPORTED == "unsupported"
    assert CitationSupportLabel.CONTRADICTED == "contradicted"
    assert CitationSupportLabel.CITATION_MISSING == "citation_missing"
    assert CitationSupportLabel.CITATION_PHANTOM == "citation_phantom"
    assert CitationSupportLabel.UNKNOWN == "unknown"


def test_citation_faithfulness_risk_values() -> None:
    assert CitationFaithfulnessRisk.LOW == "low"
    assert CitationFaithfulnessRisk.MEDIUM == "medium"
    assert CitationFaithfulnessRisk.HIGH == "high"
    assert CitationFaithfulnessRisk.UNKNOWN == "unknown"


def test_citation_evidence_source_values() -> None:
    assert CitationEvidenceSource.CLAIM_GROUNDING == "claim_grounding"
    assert CitationEvidenceSource.RETRIEVAL_EVIDENCE_PROFILE == "retrieval_evidence_profile"
    assert CitationEvidenceSource.LEGACY_CITATION_IDS == "legacy_citation_ids"
    assert CitationEvidenceSource.UNAVAILABLE == "unavailable"


def test_method_and_calibration_values() -> None:
    assert CitationMethodType.HEURISTIC_BASELINE == "heuristic_baseline"
    assert CitationMethodType.PRACTICAL_APPROXIMATION == "practical_approximation"
    assert CitationMethodType.RESEARCH_FAITHFUL == "research_faithful"
    assert CitationCalibrationStatus.UNCALIBRATED == "uncalibrated"
    assert CitationCalibrationStatus.MOCK_CALIBRATED == "mock_calibrated"
    assert CitationCalibrationStatus.CALIBRATED == "calibrated"


def test_claim_citation_faithfulness_record_defaults() -> None:
    record = ClaimCitationFaithfulnessRecord(
        claim_id="claim-1",
        claim_text="The document says the benefit is available.",
    )

    assert record.cited_doc_ids == []
    assert record.cited_chunk_ids == []
    assert record.supporting_chunk_ids == []
    assert record.contradicted_by_chunk_ids == []
    assert record.neutral_chunk_ids == []
    assert record.citation_support_label == CitationSupportLabel.UNKNOWN
    assert record.faithfulness_risk == CitationFaithfulnessRisk.UNKNOWN
    assert record.evidence_source == CitationEvidenceSource.UNAVAILABLE
    assert record.explanation is None
    assert record.warnings == []
    assert record.limitations == []


def test_citation_faithfulness_report_defaults() -> None:
    report = CitationFaithfulnessReport()

    assert report.run_id is None
    assert report.records == []
    assert report.unsupported_claim_ids == []
    assert report.phantom_citation_doc_ids == []
    assert report.missing_citation_claim_ids == []
    assert report.contradicted_claim_ids == []
    assert report.claim_grounding_used is False
    assert report.retrieval_evidence_profile_used is False
    assert report.legacy_citation_fallback_used is False
    assert report.method_type == CitationMethodType.PRACTICAL_APPROXIMATION
    assert report.calibration_status == CitationCalibrationStatus.UNCALIBRATED
    assert report.recommended_for_gating is False
    assert report.limitations == []


def test_citation_faithfulness_report_json_serialization() -> None:
    report = CitationFaithfulnessReport(
        run_id="run-1",
        records=[
            ClaimCitationFaithfulnessRecord(
                claim_id="claim-1",
                claim_text="The cited order supports the claim.",
                cited_doc_ids=["doc-1"],
                citation_support_label=CitationSupportLabel.FULLY_SUPPORTED,
                faithfulness_risk=CitationFaithfulnessRisk.LOW,
                evidence_source=CitationEvidenceSource.CLAIM_GROUNDING,
            )
        ],
    )

    dumped = report.model_dump(mode="json")

    assert dumped["run_id"] == "run-1"
    assert dumped["method_type"] == "practical_approximation"
    assert dumped["calibration_status"] == "uncalibrated"
    assert dumped["records"][0]["citation_support_label"] == "fully_supported"
    assert dumped["records"][0]["faithfulness_risk"] == "low"
    assert dumped["records"][0]["evidence_source"] == "claim_grounding"


def test_mutable_defaults_are_not_shared_across_instances() -> None:
    first_record = ClaimCitationFaithfulnessRecord(
        claim_id="claim-1",
        claim_text="First claim.",
    )
    second_record = ClaimCitationFaithfulnessRecord(
        claim_id="claim-2",
        claim_text="Second claim.",
    )
    first_record.cited_doc_ids.append("doc-1")
    first_record.warnings.append("warning")

    first_report = CitationFaithfulnessReport()
    second_report = CitationFaithfulnessReport()
    first_report.records.append(first_record)
    first_report.unsupported_claim_ids.append("claim-1")

    assert second_record.cited_doc_ids == []
    assert second_record.warnings == []
    assert second_report.records == []
    assert second_report.unsupported_claim_ids == []
