"""
Data models for temporal source validity analysis.

These models define the v1 data substrate for a future Version / Temporal
Validity Analyzer. They do not implement analyzer behavior, LLM judging,
VersionRAG, or domain-specific lifecycle reasoning.

IMPORTANT:
- v1 temporal source validity is a practical approximation.
- These models are not research-faithful domain-specific temporal reasoning.
- Reports are not recommended for production gating unless calibrated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentValidityStatus(str, Enum):
    """Temporal validity status for a source document."""

    ACTIVE = "active"
    STALE_BY_AGE = "stale_by_age"
    SUPERSEDED = "superseded"
    AMENDED = "amended"
    REPLACED = "replaced"
    DEPRECATED = "deprecated"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    APPLICABILITY_UNKNOWN = "applicability_unknown"
    METADATA_MISSING = "metadata_missing"
    UNKNOWN = "unknown"


class DocumentValidityRisk(str, Enum):
    """Risk level associated with document temporal validity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ValidityEvidenceSource(str, Enum):
    """Source used to populate version validity evidence."""

    CORPUS_METADATA = "corpus_metadata"
    RETRIEVAL_EVIDENCE_PROFILE = "retrieval_evidence_profile"
    CITATION_FAITHFULNESS_REPORT = "citation_faithfulness_report"
    DOCUMENT_LINEAGE = "document_lineage"
    HEURISTIC_AGE_CHECK = "heuristic_age_check"
    UNAVAILABLE = "unavailable"


class VersionValidityMethodType(str, Enum):
    """Epistemological tier of a temporal source validity method."""

    HEURISTIC_BASELINE = "heuristic_baseline"
    PRACTICAL_APPROXIMATION = "practical_approximation"
    RESEARCH_FAITHFUL = "research_faithful"


class VersionValidityCalibrationStatus(str, Enum):
    """Calibration state of a temporal source validity report."""

    UNCALIBRATED = "uncalibrated"
    MOCK_CALIBRATED = "mock_calibrated"
    CALIBRATED = "calibrated"


class DocumentValidityRecord(BaseModel):
    """
    Temporal validity evidence for one source document.

    This is a data record only. It can represent corpus metadata, retrieval
    evidence, citation evidence, or document lineage signals. Unknown defaults
    prevent callers from silently inferring that a document is active.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    doc_id: str
    source_doc_id: Optional[str] = None
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    department: Optional[str] = None
    version_id: Optional[str] = None
    issue_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    query_date: Optional[datetime] = None
    validity_status: DocumentValidityStatus = DocumentValidityStatus.UNKNOWN
    validity_risk: DocumentValidityRisk = DocumentValidityRisk.UNKNOWN
    supersedes_doc_ids: List[str] = Field(default_factory=list)
    superseded_by_doc_ids: List[str] = Field(default_factory=list)
    amends_doc_ids: List[str] = Field(default_factory=list)
    amended_by_doc_ids: List[str] = Field(default_factory=list)
    replaces_doc_ids: List[str] = Field(default_factory=list)
    replaced_by_doc_ids: List[str] = Field(default_factory=list)
    deprecated_by_doc_ids: List[str] = Field(default_factory=list)
    withdrawn_by_doc_ids: List[str] = Field(default_factory=list)
    evidence_source: ValidityEvidenceSource = ValidityEvidenceSource.UNAVAILABLE
    evidence_paths: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ClaimSourceValidityRecord(BaseModel):
    """
    Claim-level view of cited source temporal validity.

    This record links a claim to cited documents and summarizes whether invalid
    or unknown source validity may affect that claim. It does not itself prove
    domain-specific applicability or temporal correctness.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_id: str
    claim_text: Optional[str] = None
    cited_doc_ids: List[str] = Field(default_factory=list)
    invalid_cited_doc_ids: List[str] = Field(default_factory=list)
    valid_cited_doc_ids: List[str] = Field(default_factory=list)
    unknown_validity_doc_ids: List[str] = Field(default_factory=list)
    claim_validity_status: DocumentValidityStatus = DocumentValidityStatus.UNKNOWN
    claim_validity_risk: DocumentValidityRisk = DocumentValidityRisk.UNKNOWN
    does_invalid_source_affect_claim: Optional[bool] = None
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    evidence_paths: List[str] = Field(default_factory=list)


class VersionValidityReport(BaseModel):
    """
    Run-level version and temporal validity report.

    v1 is a practical approximation, not research-faithful VersionRAG or
    domain-specific temporal reasoning. The report is uncalibrated by default
    and must not be treated as production gating guidance unless future
    calibration explicitly changes that contract.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: Optional[str] = None
    query_date: Optional[datetime] = None
    document_records: List[DocumentValidityRecord] = Field(default_factory=list)
    claim_source_records: List[ClaimSourceValidityRecord] = Field(default_factory=list)
    active_doc_ids: List[str] = Field(default_factory=list)
    stale_doc_ids: List[str] = Field(default_factory=list)
    superseded_doc_ids: List[str] = Field(default_factory=list)
    amended_doc_ids: List[str] = Field(default_factory=list)
    withdrawn_doc_ids: List[str] = Field(default_factory=list)
    replaced_doc_ids: List[str] = Field(default_factory=list)
    deprecated_doc_ids: List[str] = Field(default_factory=list)
    expired_doc_ids: List[str] = Field(default_factory=list)
    not_yet_effective_doc_ids: List[str] = Field(default_factory=list)
    metadata_missing_doc_ids: List[str] = Field(default_factory=list)
    high_risk_claim_ids: List[str] = Field(default_factory=list)
    cited_invalid_doc_ids: List[str] = Field(default_factory=list)
    answer_bearing_invalid_doc_ids: List[str] = Field(default_factory=list)
    retrieved_only_stale_doc_ids: List[str] = Field(default_factory=list)
    stale_but_irrelevant_doc_ids: List[str] = Field(default_factory=list)
    retrieval_quality_affected_doc_ids: List[str] = Field(default_factory=list)
    retrieval_evidence_profile_used: bool = False
    citation_faithfulness_report_used: bool = False
    lineage_metadata_used: bool = False
    age_based_fallback_used: bool = False
    method_type: VersionValidityMethodType = (
        VersionValidityMethodType.PRACTICAL_APPROXIMATION
    )
    calibration_status: VersionValidityCalibrationStatus = (
        VersionValidityCalibrationStatus.UNCALIBRATED
    )
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)
