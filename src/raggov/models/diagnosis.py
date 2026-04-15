"""Data models describing analyzer diagnoses and findings."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FailureStage(str, Enum):
    """Pipeline stages where a RagGov failure can originate."""

    RETRIEVAL = "RETRIEVAL"
    GROUNDING = "GROUNDING"
    SUFFICIENCY = "SUFFICIENCY"
    SECURITY = "SECURITY"
    CONFIDENCE = "CONFIDENCE"
    UNKNOWN = "UNKNOWN"


class FailureType(str, Enum):
    """Known RagGov failure categories."""

    STALE_RETRIEVAL = "STALE_RETRIEVAL"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    CITATION_MISMATCH = "CITATION_MISMATCH"
    INCONSISTENT_CHUNKS = "INCONSISTENT_CHUNKS"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    CONTRADICTED_CLAIM = "CONTRADICTED_CLAIM"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    SUSPICIOUS_CHUNK = "SUSPICIOUS_CHUNK"
    RETRIEVAL_ANOMALY = "RETRIEVAL_ANOMALY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CLEAN = "CLEAN"


class SecurityRisk(str, Enum):
    """Security risk levels for a RagGov diagnosis."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ClaimResult(BaseModel):
    """Grounding result for a single answer claim."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_text: str
    label: Literal["entailed", "unsupported", "contradicted"]
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class AnalyzerResult(BaseModel):
    """Result emitted by an individual RagGov analyzer."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    analyzer_name: str
    status: Literal["pass", "warn", "fail", "skip"]
    failure_type: FailureType | None = None
    stage: FailureStage | None = None
    score: float | None = None
    security_risk: SecurityRisk | None = None
    evidence: list[str] = Field(default_factory=list)
    remediation: str | None = None


class Diagnosis(BaseModel):
    """Overall diagnosis for a RagGov run."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    primary_failure: FailureType
    secondary_failures: list[FailureType] = Field(default_factory=list)
    root_cause_stage: FailureStage
    should_have_answered: bool
    security_risk: SecurityRisk
    confidence: float | None
    claim_results: list[ClaimResult] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommended_fix: str
    checks_run: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)
    analyzer_results: list[AnalyzerResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> str:
        """Return a single human-readable paragraph summarizing the diagnosis."""
        return (
            f"Run {self.run_id}: {self.primary_failure.value} at "
            f"{self.root_cause_stage.value} stage. Should have answered: "
            f"{self.should_have_answered}. Security risk: {self.security_risk.value}. "
            f"Recommended fix: {self.recommended_fix}"
        )
