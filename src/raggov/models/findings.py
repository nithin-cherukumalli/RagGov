"""Data models for analyzer findings and reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.signals import EvidenceSignalMetadata


class AnalyzerFinding(BaseModel):
    """A single atomic verification outcome or engineering defect found by an analyzer."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    finding_id: str
    analyzer_name: str
    failure_type: FailureType | None
    stage: FailureStage | None
    status: Literal["pass", "warn", "fail", "skip"]
    severity: Literal["none", "low", "medium", "high", "critical"]
    evidence_message: str
    signal_metadata: EvidenceSignalMetadata
    affected_chunk_ids: list[str] = Field(default_factory=list)
    affected_doc_ids: list[str] = Field(default_factory=list)
    affected_claim_ids: list[str] = Field(default_factory=list)


class AnalyzerReport(BaseModel):
    """Structured output from an individual analyzer aggregating multiple findings."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    analyzer_name: str
    overall_status: Literal["pass", "warn", "fail", "skip"]
    findings: list[AnalyzerFinding] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_heuristics_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    elapsed_ms: float | None = None
