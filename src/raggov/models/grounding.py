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

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimVerificationLabel(str, Enum):
    """ Standardized labels for claim verification status. """
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"
    UNVERIFIED = "unverified"


def normalize_claim_verification_label(raw: str | ClaimVerificationLabel) -> ClaimVerificationLabel:
    """Normalize verifier label aliases to GovRAG's internal claim label enum."""
    if isinstance(raw, ClaimVerificationLabel):
        return raw

    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entailed": ClaimVerificationLabel.ENTAILED,
        "support": ClaimVerificationLabel.ENTAILED,
        "supported": ClaimVerificationLabel.ENTAILED,
        "supports": ClaimVerificationLabel.ENTAILED,
        "contradicted": ClaimVerificationLabel.CONTRADICTED,
        "contradiction": ClaimVerificationLabel.CONTRADICTED,
        "contradicts": ClaimVerificationLabel.CONTRADICTED,
        "unsupported": ClaimVerificationLabel.INSUFFICIENT,
        "insufficient": ClaimVerificationLabel.INSUFFICIENT,
        "not_supported": ClaimVerificationLabel.INSUFFICIENT,
        "unverified": ClaimVerificationLabel.UNVERIFIED,
        "unknown": ClaimVerificationLabel.NEUTRAL,
        "unclear": ClaimVerificationLabel.NEUTRAL,
        "neutral": ClaimVerificationLabel.NEUTRAL,
        "abstain": ClaimVerificationLabel.NEUTRAL,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Invalid claim verification label {raw!r}. Expected one of: {valid}"
        ) from exc


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
    source_sentence: Optional[str] = None
    source_answer_span: Optional[Tuple[int, int]] = None
    source_answer_id: Optional[str] = None
    atomicity_status: Optional[str] = None
    claim_type: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    numbers: List[str] = Field(default_factory=list)
    extraction_method: Optional[str] = None
    extraction_reason: Optional[str] = None
    extraction_confidence: Optional[float] = None
    extraction_warnings: List[str] = Field(default_factory=list)
    skip_reason: Optional[str] = None
    
    # Evidence Linkages
    cited_doc_ids: List[str] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    candidate_evidence_chunks: List[Any] = Field(default_factory=list)
    candidate_evidence_chunk_ids: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    contradicting_chunk_ids: List[str] = Field(default_factory=list)
    best_candidate_id: Optional[str] = None
    best_supporting_doc_id: Optional[str] = None
    supporting_candidate_ids: List[str] = Field(default_factory=list)
    contradicting_candidate_ids: List[str] = Field(default_factory=list)
    neutral_candidate_ids: List[str] = Field(default_factory=list)
    evidence_mode: Optional[str] = None
    support_source_type: Optional[str] = None
    
    # Generic bucket for evidence references (can hold full objects or pointers)
    evidence_refs: Dict[str, Any] = Field(default_factory=dict)
    
    # Verification Metadata
    verification_label: ClaimVerificationLabel = ClaimVerificationLabel.UNVERIFIED
    support_label: Optional[str] = None
    support_reason: Optional[str] = None
    verifier_method: str = "unknown"
    verifier_score: float = 0.0
    raw_support_score: float = 0.0
    label_reason: str | None = None
    calibrated_confidence: Optional[float] = None
    confidence_status: Optional[str] = None
    verifier_limitations: List[str] = Field(default_factory=list)
    verifier_warnings: List[str] = Field(default_factory=list)
    raw_entailment_response: Any = None
    fallback_from: Optional[str] = None
    fallback_to: Optional[str] = None
    value_matches: List[Dict[str, str]] = Field(default_factory=list)
    value_conflicts: List[Dict[str, str]] = Field(default_factory=list)
    
    # Safety Gate and Ensemble Fields
    verifier_policy: Optional[str] = None
    verifier_disagreement: bool = False
    safety_gate_triggered: bool = False
    safety_gate_reason: Optional[str] = None
    safety_gate_category: Optional[str] = None
    critical_fact_check_summary: Dict[str, Any] = Field(default_factory=dict)
    llm_label: Optional[str] = None
    heuristic_label: Optional[str] = None
    deterministic_gate_labels: List[str] = Field(default_factory=list)
    normalized_values_checked: List[Dict[str, Any]] = Field(default_factory=list)
    normalized_dates_checked: List[Dict[str, Any]] = Field(default_factory=list)
    normalized_units_checked: List[Dict[str, Any]] = Field(default_factory=list)
    normalized_entities_checked: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Calibration and Trust
    calibration_status: CalibrationStatus = CalibrationStatus.UNAVAILABLE
    uncertainty_signals: Dict[str, Any] = Field(default_factory=dict)
    external_signal_records: List[Dict[str, Any]] = Field(default_factory=list)
    
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

    @field_validator("verification_label", mode="before")
    @classmethod
    def _normalize_verification_label(cls, value: object) -> ClaimVerificationLabel:
        return normalize_claim_verification_label(value)  # type: ignore[arg-type]

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
    external_signal_records: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
