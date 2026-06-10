"""Shared metadata contract for diagnosis-relevant evidence signals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MethodStatus = Literal[
    "structured_deterministic",
    "heuristic_baseline",
    "practical_approximation",
    "external_advisory",
    "calibrated_statistical",
]

SignalCalibrationStatus = Literal[
    "uncalibrated",
    "provisional_dataset",
    "calibrated_dev",
    "calibrated_heldout",
    "unknown",
]

EvidenceStrength = Literal[
    "hard",
    "strong",
    "medium",
    "weak",
    "advisory",
]

SignalEvidenceTier = Literal[
    "structured",
    "heuristic",
    "proxy",
    "external",
    "calibrated",
]


class EvidenceSignalMetadata(BaseModel):
    """Calibration and strength metadata for one diagnosis-relevant signal."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    signal_name: str
    source_analyzer: str
    method: str
    method_status: MethodStatus
    calibration_status: SignalCalibrationStatus
    evidence_strength: EvidenceStrength
    evidence_tier: SignalEvidenceTier
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class RetrievalEvidenceMetadata(EvidenceSignalMetadata):
    """Audit-friendly metadata for retrieval ownership evidence."""

    stage: str = "RETRIEVAL"
    retrieved_k: int | None = None
    configured_top_k: int | None = None
    candidate_pool_size: int | None = None
    min_required_k: int | None = None
    has_more_candidates: bool | None = None
    relevant_hits_at_k: int | None = None
    relevant_hits_beyond_k: int | None = None
    top_k_relevance_distribution: list[float] = Field(default_factory=list)
    scope_match_score: float | None = None
    scope_drift_score: float | None = None
    retrieval_anomaly_score: float | None = None
    reason: str | None = None
    query_scope_terms: list[str] = Field(default_factory=list)
    retrieved_scope_terms: list[str] = Field(default_factory=list)
    missing_required_scope_terms: list[str] = Field(default_factory=list)
    scope_drift_reason: str | None = None
    anomaly_type: str | None = None
    anomalous_chunk_ids: list[str] = Field(default_factory=list)
    rank_position: int | None = None
    source_trust_flags: list[str] = Field(default_factory=list)
    similarity_outlier: bool | None = None
    metadata_inconsistency: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def default_uncalibrated_heuristic_signal(
    signal_name: str,
    source_analyzer: str,
    method: str = "unknown",
    notes: str | None = None,
) -> EvidenceSignalMetadata:
    """Return the conservative default for signals without explicit metadata."""

    return EvidenceSignalMetadata(
        signal_name=signal_name,
        source_analyzer=source_analyzer,
        method=method,
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
        notes=notes,
    )


__all__ = [
    "EvidenceSignalMetadata",
    "EvidenceStrength",
    "MethodStatus",
    "RetrievalEvidenceMetadata",
    "SignalCalibrationStatus",
    "SignalEvidenceTier",
    "default_uncalibrated_heuristic_signal",
]
