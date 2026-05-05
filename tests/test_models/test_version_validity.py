"""Tests for version validity data models."""

from __future__ import annotations

from datetime import UTC, datetime

from raggov.models import (
    ClaimSourceValidityRecord,
    DocumentValidityRecord,
    DocumentValidityRisk,
    DocumentValidityStatus,
    ValidityEvidenceSource,
    VersionValidityCalibrationStatus,
    VersionValidityMethodType,
    VersionValidityReport,
)


def test_document_validity_status_values() -> None:
    assert DocumentValidityStatus.ACTIVE == "active"
    assert DocumentValidityStatus.STALE_BY_AGE == "stale_by_age"
    assert DocumentValidityStatus.SUPERSEDED == "superseded"
    assert DocumentValidityStatus.AMENDED == "amended"
    assert DocumentValidityStatus.REPLACED == "replaced"
    assert DocumentValidityStatus.DEPRECATED == "deprecated"
    assert DocumentValidityStatus.WITHDRAWN == "withdrawn"
    assert DocumentValidityStatus.EXPIRED == "expired"
    assert DocumentValidityStatus.NOT_YET_EFFECTIVE == "not_yet_effective"
    assert DocumentValidityStatus.APPLICABILITY_UNKNOWN == "applicability_unknown"
    assert DocumentValidityStatus.METADATA_MISSING == "metadata_missing"
    assert DocumentValidityStatus.UNKNOWN == "unknown"


def test_document_validity_risk_values() -> None:
    assert DocumentValidityRisk.LOW == "low"
    assert DocumentValidityRisk.MEDIUM == "medium"
    assert DocumentValidityRisk.HIGH == "high"
    assert DocumentValidityRisk.UNKNOWN == "unknown"


def test_validity_evidence_source_values() -> None:
    assert ValidityEvidenceSource.CORPUS_METADATA == "corpus_metadata"
    assert ValidityEvidenceSource.RETRIEVAL_EVIDENCE_PROFILE == "retrieval_evidence_profile"
    assert ValidityEvidenceSource.CITATION_FAITHFULNESS_REPORT == "citation_faithfulness_report"
    assert ValidityEvidenceSource.DOCUMENT_LINEAGE == "document_lineage"
    assert ValidityEvidenceSource.HEURISTIC_AGE_CHECK == "heuristic_age_check"
    assert ValidityEvidenceSource.UNAVAILABLE == "unavailable"


def test_method_and_calibration_values() -> None:
    assert VersionValidityMethodType.HEURISTIC_BASELINE == "heuristic_baseline"
    assert VersionValidityMethodType.PRACTICAL_APPROXIMATION == "practical_approximation"
    assert VersionValidityMethodType.RESEARCH_FAITHFUL == "research_faithful"
    assert VersionValidityCalibrationStatus.UNCALIBRATED == "uncalibrated"
    assert VersionValidityCalibrationStatus.MOCK_CALIBRATED == "mock_calibrated"
    assert VersionValidityCalibrationStatus.CALIBRATED == "calibrated"


def test_document_validity_record_defaults() -> None:
    record = DocumentValidityRecord(doc_id="doc-1")

    assert record.source_doc_id is None
    assert record.document_title is None
    assert record.document_type is None
    assert record.department is None
    assert record.version_id is None
    assert record.issue_date is None
    assert record.effective_date is None
    assert record.expiry_date is None
    assert record.query_date is None
    assert record.validity_status == DocumentValidityStatus.UNKNOWN
    assert record.validity_risk == DocumentValidityRisk.UNKNOWN
    assert record.supersedes_doc_ids == []
    assert record.superseded_by_doc_ids == []
    assert record.amends_doc_ids == []
    assert record.amended_by_doc_ids == []
    assert record.replaces_doc_ids == []
    assert record.replaced_by_doc_ids == []
    assert record.deprecated_by_doc_ids == []
    assert record.withdrawn_by_doc_ids == []
    assert record.evidence_paths == []
    assert record.evidence_source == ValidityEvidenceSource.UNAVAILABLE
    assert record.explanation is None
    assert record.warnings == []
    assert record.evidence_paths == []
    assert record.limitations == []


def test_claim_source_validity_record_defaults() -> None:
    record = ClaimSourceValidityRecord(claim_id="claim-1")

    assert record.claim_text is None
    assert record.cited_doc_ids == []
    assert record.invalid_cited_doc_ids == []
    assert record.valid_cited_doc_ids == []
    assert record.unknown_validity_doc_ids == []
    assert record.claim_validity_status == DocumentValidityStatus.UNKNOWN
    assert record.claim_validity_risk == DocumentValidityRisk.UNKNOWN
    assert record.does_invalid_source_affect_claim is None
    assert record.explanation is None
    assert record.warnings == []


def test_version_validity_report_defaults() -> None:
    report = VersionValidityReport()

    assert report.run_id is None
    assert report.query_date is None
    assert report.document_records == []
    assert report.claim_source_records == []
    assert report.active_doc_ids == []
    assert report.stale_doc_ids == []
    assert report.superseded_doc_ids == []
    assert report.amended_doc_ids == []
    assert report.withdrawn_doc_ids == []
    assert report.replaced_doc_ids == []
    assert report.deprecated_doc_ids == []
    assert report.expired_doc_ids == []
    assert report.not_yet_effective_doc_ids == []
    assert report.metadata_missing_doc_ids == []
    assert report.high_risk_claim_ids == []
    assert report.retrieval_evidence_profile_used is False
    assert report.citation_faithfulness_report_used is False
    assert report.lineage_metadata_used is False
    assert report.age_based_fallback_used is False
    assert report.method_type == VersionValidityMethodType.PRACTICAL_APPROXIMATION
    assert report.calibration_status == VersionValidityCalibrationStatus.UNCALIBRATED
    assert report.recommended_for_gating is False
    assert report.limitations == []


def test_version_validity_report_json_serialization() -> None:
    query_date = datetime(2026, 5, 2, tzinfo=UTC)
    report = VersionValidityReport(
        run_id="run-1",
        query_date=query_date,
        document_records=[
            DocumentValidityRecord(
                doc_id="doc-1",
                issue_date=datetime(2026, 1, 1, tzinfo=UTC),
                version_id="v1",
                validity_status=DocumentValidityStatus.ACTIVE,
                validity_risk=DocumentValidityRisk.LOW,
                evidence_source=ValidityEvidenceSource.CORPUS_METADATA,
                evidence_paths=["metadata.status"],
            )
        ],
        claim_source_records=[
            ClaimSourceValidityRecord(
                claim_id="claim-1",
                cited_doc_ids=["doc-1"],
                valid_cited_doc_ids=["doc-1"],
                claim_validity_status=DocumentValidityStatus.ACTIVE,
                claim_validity_risk=DocumentValidityRisk.LOW,
                evidence_paths=["claim.cited_doc_ids"],
            )
        ],
    )

    dumped = report.model_dump(mode="json")

    assert dumped["run_id"] == "run-1"
    assert dumped["query_date"] == "2026-05-02T00:00:00Z"
    assert dumped["document_records"][0]["issue_date"] == "2026-01-01T00:00:00Z"
    assert dumped["document_records"][0]["version_id"] == "v1"
    assert dumped["document_records"][0]["validity_status"] == "active"
    assert dumped["document_records"][0]["validity_risk"] == "low"
    assert dumped["document_records"][0]["evidence_source"] == "corpus_metadata"
    assert dumped["document_records"][0]["evidence_paths"] == ["metadata.status"]
    assert dumped["claim_source_records"][0]["claim_validity_status"] == "active"
    assert dumped["method_type"] == "practical_approximation"
    assert dumped["calibration_status"] == "uncalibrated"


def test_mutable_defaults_are_not_shared_across_instances() -> None:
    first_doc = DocumentValidityRecord(doc_id="doc-1")
    second_doc = DocumentValidityRecord(doc_id="doc-2")
    first_doc.superseded_by_doc_ids.append("doc-new")
    first_doc.warnings.append("warning")
    first_doc.evidence_paths.append("metadata.status")

    first_claim = ClaimSourceValidityRecord(claim_id="claim-1")
    second_claim = ClaimSourceValidityRecord(claim_id="claim-2")
    first_claim.invalid_cited_doc_ids.append("doc-1")

    first_report = VersionValidityReport()
    second_report = VersionValidityReport()
    first_report.document_records.append(first_doc)
    first_report.high_risk_claim_ids.append("claim-1")

    assert second_doc.superseded_by_doc_ids == []
    assert second_doc.warnings == []
    assert second_doc.evidence_paths == []
    assert second_claim.invalid_cited_doc_ids == []
    assert second_report.document_records == []
    assert second_report.high_risk_claim_ids == []
