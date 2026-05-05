"""
Domain-agnostic retrieval-stage diagnosis models.

These models represent a thin rollup over existing retrieval, sufficiency,
grounding, citation, and temporal-validity evidence. v0 is a heuristic baseline,
uncalibrated, and not recommended for production gating.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class RetrievalFailureType(str, Enum):
    """Likely retrieval-stage failure mode inferred from upstream evidence."""

    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_NOISE = "retrieval_noise"
    RANK_FAILURE_UNKNOWN = "rank_failure_unknown"
    VERSION_RETRIEVAL_FAILURE = "version_retrieval_failure"
    CITATION_RETRIEVAL_MISMATCH = "citation_retrieval_mismatch"
    NO_CLEAR_RETRIEVAL_FAILURE = "no_clear_retrieval_failure"
    INSUFFICIENT_EVIDENCE_TO_DIAGNOSE = "insufficient_evidence_to_diagnose"


class RetrievalDiagnosisMethodType(str, Enum):
    """Epistemological tier of the retrieval diagnosis method."""

    HEURISTIC_BASELINE = "heuristic_baseline"
    PRACTICAL_APPROXIMATION = "practical_approximation"


class RetrievalDiagnosisCalibrationStatus(str, Enum):
    """Calibration state of the retrieval diagnosis report."""

    UNCALIBRATED = "uncalibrated"
    PRELIMINARY_CALIBRATED = "preliminary_calibrated"
    CALIBRATED = "calibrated"


class RetrievalEvidenceSignal(BaseModel):
    """One upstream evidence signal used by the retrieval diagnosis rollup."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    signal_name: str
    value: str | float | int | bool | None
    source_report: str | None
    source_ids: List[str] = Field(default_factory=list)
    interpretation: str
    limitation: str | None = None


class ClaimRetrievalDiagnosisRecord(BaseModel):
    """Claim-level retrieval diagnosis assembled from existing evidence."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_id: str
    claim_text: str | None = None
    retrieval_failure_type: RetrievalFailureType | None = None
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    candidate_chunk_ids: List[str] = Field(default_factory=list)
    noisy_chunk_ids: List[str] = Field(default_factory=list)
    invalid_source_doc_ids: List[str] = Field(default_factory=list)
    evidence_signals: List[RetrievalEvidenceSignal] = Field(default_factory=list)
    explanation: str


class RetrievalDiagnosisReport(BaseModel):
    """Run-level retrieval-stage diagnosis rollup."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    primary_failure_type: RetrievalFailureType
    affected_claim_ids: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    candidate_chunk_ids: List[str] = Field(default_factory=list)
    noisy_chunk_ids: List[str] = Field(default_factory=list)
    invalid_retrieved_doc_ids: List[str] = Field(default_factory=list)
    invalid_cited_doc_ids: List[str] = Field(default_factory=list)
    missing_reports: List[str] = Field(default_factory=list)
    fallback_heuristics_used: List[str] = Field(default_factory=list)
    claim_records: List[ClaimRetrievalDiagnosisRecord] = Field(default_factory=list)
    evidence_signals: List[RetrievalEvidenceSignal] = Field(default_factory=list)
    alternative_explanations: List[str] = Field(default_factory=list)
    recommended_fix: str
    method_type: RetrievalDiagnosisMethodType
    calibration_status: RetrievalDiagnosisCalibrationStatus
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)
