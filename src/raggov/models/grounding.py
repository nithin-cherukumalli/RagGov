"""
Domain-agnostic data models for claim evidence and verification.

These models provide a shared substrate for claim grounding, citation
faithfulness, sufficiency, and downstream classification/attribution tasks.
They are designed to be neutral across domains (legal, medical, policy, etc.).

IMPORTANT:
- These models do not themselves prove claim correctness.
- Verification labels MUST come from a verifier implementation, not extraction.
- Heuristic outputs must be clearly marked as heuristic or uncalibrated.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class ClaimVerificationLabel(str, Enum):
    """ Standardized labels for claim verification status. """
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"
    UNVERIFIED = "unverified"


class CalibrationStatus(str, Enum):
    """ Status of the verification confidence/score calibration. """
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


class StructuredClaimRepresentation(BaseModel):
    """
    Flexible container for structured representations of a claim.
    
    This model allows optional representations such as triplets, frames,
    events, relations, or QA pairs. It does not assume a specific domain
    or structure (e.g., it is not restricted to subject-predicate-object).
    """
    model_config = ConfigDict(extra="allow")

    triplet: Optional[Dict[str, Any]] = None
    frame: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    relation: Optional[Dict[str, Any]] = None
    qa_pair: Optional[Dict[str, Any]] = None
    
    # Generic bucket for other representations
    extra_representations: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimEvidenceRecord(BaseModel):
    """
    Domain-agnostic record linking a claim to its evidence and verification metadata.
    
    This model is the primary data structure for claim-level analysis in GovRAG.
    It tracks the lifecycle of a claim from extraction through verification
    to final attribution.
    """
    model_config = ConfigDict(frozen=False, extra="allow")

    # Claim Identification
    claim_id: str
    claim_text: str
    source_answer_span: Optional[Tuple[int, int]] = None
    source_answer_id: Optional[str] = None
    
    # Evidence Linkages
    cited_chunk_ids: List[str] = Field(default_factory=list)
    candidate_evidence_chunk_ids: List[str] = Field(default_factory=list)
    
    # Generic bucket for evidence references (can hold full objects or pointers)
    evidence_refs: Dict[str, Any] = Field(default_factory=dict)
    
    # Verification Metadata
    verification_label: ClaimVerificationLabel = ClaimVerificationLabel.UNVERIFIED
    verifier_method: str = "unknown"
    verifier_score: float = 0.0
    
    # Calibration and Trust
    calibration_status: CalibrationStatus = CalibrationStatus.UNAVAILABLE
    uncertainty_signals: Dict[str, Any] = Field(default_factory=dict)
    
    # Provenance and Context
    provenance: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    
    # Domain-Specific Data
    # Used for outputs from domain-specific adapters (e.g., policy extractor)
    domain_adapter_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Structural representations (e.g., RefChecker triplets)
    structured_representation: Optional[StructuredClaimRepresentation] = None
    
    # Audit trail
    created_by: str = "system"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GroundingEvidenceBundle(BaseModel):
    """
    Stable interface for sharing structured grounding evidence with downstream analyzers.
    
    This bundle aggregates claim-level records, diagnostic rollups, and
    calibration summaries into a single portable object.
    """
    model_config = ConfigDict(frozen=False, extra="allow")

    claim_evidence_records: List[ClaimEvidenceRecord] = Field(default_factory=list)
    diagnostic_rollup: Optional[Dict[str, Any]] = None
    citation_support_summary: Dict[str, Any] = Field(default_factory=dict)
    calibration_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
