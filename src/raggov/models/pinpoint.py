"""Structured pinpointing data models for NCV+A2P failure localization.

These models represent a shared substrate for structured failure pinpointing.
All fields are heuristic/uncalibrated by default.  No production-gating
decisions are derived from these models; recommended_for_gating defaults to
False and calibrated_confidence is always optional/nullable.
"""

from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PinpointLocation(BaseModel):
    """Structured description of where a failure was localized."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    location_id: str
    pipeline_stage: str | None = None
    ncv_node: str | None = None
    failure_type: str | None = None
    claim_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    citation_ids: List[str] = Field(default_factory=list)
    doc_ids: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    localization_method: str
    method_type: str
    calibration_status: str
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)


class PinpointEvidence(BaseModel):
    """One evidence signal supporting or opposing a pinpoint finding."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    signal_name: str
    value: Any = None
    label: str | None = None
    source_report: str
    provider: str | None = None
    affected_claim_ids: List[str] = Field(default_factory=list)
    affected_chunk_ids: List[str] = Field(default_factory=list)
    affected_doc_ids: List[str] = Field(default_factory=list)
    affected_citation_ids: List[str] = Field(default_factory=list)
    interpretation: str
    method_type: str
    calibration_status: str
    limitations: List[str] = Field(default_factory=list)


class PinpointFinding(BaseModel):
    """Structured failure finding produced by NCV or A2P pinpointing.

    calibrated_confidence is always None until a calibration workflow exists.
    recommended_for_gating defaults to False; do not set True without a
    validated calibration pass.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    finding_id: str
    location: PinpointLocation
    evidence_for: List[PinpointEvidence] = Field(default_factory=list)
    evidence_against: List[PinpointEvidence] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    fallback_heuristics_used: List[str] = Field(default_factory=list)
    alternative_locations: List[PinpointLocation] = Field(default_factory=list)
    heuristic_score: float | None = None
    calibrated_confidence: float | None = None
    calibration_status: str = "uncalibrated"
    human_review_recommended: bool = True
    recommended_for_gating: bool = False


class CausalChain(BaseModel):
    """Causal chain linking a root failure location to downstream effects."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    chain_id: str
    root_location: PinpointLocation
    downstream_locations: List[PinpointLocation] = Field(default_factory=list)
    causal_hypothesis: str
    abduct: str
    act: str
    predict: str
    evidence_for: List[PinpointEvidence] = Field(default_factory=list)
    evidence_against: List[PinpointEvidence] = Field(default_factory=list)
    alternative_explanations: List[str] = Field(default_factory=list)
    heuristic_score: float | None = None
    calibrated_confidence: float | None = None
    calibration_status: str = "uncalibrated"


class TrustDecision(BaseModel):
    """Structured trust/gating decision for a pinpoint finding.

    blocking_eligible and recommended_for_gating are informational only;
    no automated blocking is applied by this model.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    decision: Literal["pass", "warn", "human_review", "block"]
    reason: str
    recommended_for_gating: bool
    human_review_required: bool
    blocking_eligible: bool
    unmet_requirements: List[str] = Field(default_factory=list)
    calibration_status: str
    fallback_heuristics_used: List[str] = Field(default_factory=list)
