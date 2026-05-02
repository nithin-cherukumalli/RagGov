"""
Data models for citation faithfulness analysis in GovRAG.

These models define the v0/v1 data substrate for a future Citation
Faithfulness Analyzer. They do not implement analyzer behavior, LLM judging,
or provider-specific logic.

IMPORTANT:
- v0 citation faithfulness is a practical approximation.
- These models are not a research-faithful RefChecker or RAGChecker
  implementation.
- Reports are not recommended for production gating unless calibrated.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CitationSupportLabel(str, Enum):
    """How well the cited evidence supports a single claim."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    CITATION_MISSING = "citation_missing"
    CITATION_PHANTOM = "citation_phantom"
    UNKNOWN = "unknown"


class CitationFaithfulnessRisk(str, Enum):
    """Risk that the citation relationship is unfaithful or misleading."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class CitationEvidenceSource(str, Enum):
    """Source used to populate citation faithfulness evidence fields."""

    CLAIM_GROUNDING = "claim_grounding"
    RETRIEVAL_EVIDENCE_PROFILE = "retrieval_evidence_profile"
    LEGACY_CITATION_IDS = "legacy_citation_ids"
    UNAVAILABLE = "unavailable"


class CitationMethodType(str, Enum):
    """Epistemological tier of a citation faithfulness method."""

    HEURISTIC_BASELINE = "heuristic_baseline"
    PRACTICAL_APPROXIMATION = "practical_approximation"
    RESEARCH_FAITHFUL = "research_faithful"


class CitationCalibrationStatus(str, Enum):
    """Calibration state of a citation faithfulness report."""

    UNCALIBRATED = "uncalibrated"
    MOCK_CALIBRATED = "mock_calibrated"
    CALIBRATED = "calibrated"


class ClaimCitationFaithfulnessRecord(BaseModel):
    """
    Citation faithfulness evidence for a single generated claim.

    This is a data record only. It may be populated later from claim grounding,
    retrieval evidence profiles, or legacy citation IDs. The default state is
    unknown/unavailable so callers do not silently infer support.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_id: str
    claim_text: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    contradicted_by_chunk_ids: List[str] = Field(default_factory=list)
    neutral_chunk_ids: List[str] = Field(default_factory=list)
    citation_support_label: CitationSupportLabel = CitationSupportLabel.UNKNOWN
    faithfulness_risk: CitationFaithfulnessRisk = CitationFaithfulnessRisk.UNKNOWN
    evidence_source: CitationEvidenceSource = CitationEvidenceSource.UNAVAILABLE
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class CitationFaithfulnessReport(BaseModel):
    """
    Run-level citation faithfulness report.

    v0 is a practical approximation, not a research-faithful RefChecker or
    RAGChecker implementation. The report is uncalibrated by default and must
    not be treated as production gating guidance unless future calibration
    explicitly changes that contract.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: Optional[str] = None
    records: List[ClaimCitationFaithfulnessRecord] = Field(default_factory=list)
    unsupported_claim_ids: List[str] = Field(default_factory=list)
    phantom_citation_doc_ids: List[str] = Field(default_factory=list)
    missing_citation_claim_ids: List[str] = Field(default_factory=list)
    contradicted_claim_ids: List[str] = Field(default_factory=list)
    claim_grounding_used: bool = False
    retrieval_evidence_profile_used: bool = False
    legacy_citation_fallback_used: bool = False
    method_type: CitationMethodType = CitationMethodType.PRACTICAL_APPROXIMATION
    calibration_status: CitationCalibrationStatus = (
        CitationCalibrationStatus.UNCALIBRATED
    )
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)
