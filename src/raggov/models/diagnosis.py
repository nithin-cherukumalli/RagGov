"""Data models describing analyzer diagnoses and findings."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FailureStage(str, Enum):
    """Pipeline stages where a RagGov failure can originate."""

    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    RETRIEVAL = "RETRIEVAL"
    RERANKING = "RERANKING"
    GROUNDING = "GROUNDING"
    SUFFICIENCY = "SUFFICIENCY"
    GENERATION = "GENERATION"
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
    PRIVACY_VIOLATION = "PRIVACY_VIOLATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TABLE_STRUCTURE_LOSS = "TABLE_STRUCTURE_LOSS"
    HIERARCHY_FLATTENING = "HIERARCHY_FLATTENING"
    METADATA_LOSS = "METADATA_LOSS"
    POST_RATIONALIZED_CITATION = "POST_RATIONALIZED_CITATION"
    PARSER_STRUCTURE_LOSS = "PARSER_STRUCTURE_LOSS"
    CHUNKING_BOUNDARY_ERROR = "CHUNKING_BOUNDARY_ERROR"
    EMBEDDING_DRIFT = "EMBEDDING_DRIFT"
    RETRIEVAL_DEPTH_LIMIT = "RETRIEVAL_DEPTH_LIMIT"
    RERANKER_FAILURE = "RERANKER_FAILURE"
    GENERATION_IGNORE = "GENERATION_IGNORE"
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
    attribution_stage: FailureStage | None = None
    proposed_fix: str | None = None
    fix_confidence: float | None = None
    citation_probe_results: list[dict[str, Any]] | None = None


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
    root_cause_attribution: str | None = None
    proposed_fix: str | None = None
    fix_confidence: float | None = None
    layer6_report: dict[str, Any] | None = None
    ncv_report: dict[str, Any] | None = None
    pipeline_health_score: float | None = None
    first_failing_node: str | None = None
    citation_faithfulness: str | None = None
    failure_chain: list[str] = Field(default_factory=list)
    semantic_entropy: float | None = None

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the diagnosis.

        Format:
            Run {run_id} | {primary_failure} | Stage: {root_cause_stage}
            Should answer: {should_have_answered} | Risk: {security_risk} | Confidence: {confidence}
            Failure chain: {failure_chain}
            Root cause: {root_cause_attribution}
            Fix: {proposed_fix or recommended_fix}
        """
        lines = []

        # Line 1: Run ID, failure type, and stage
        confidence_str = f"{self.confidence:.2f}" if self.confidence is not None else "N/A"
        line1 = (
            f"Run {self.run_id} | {self.primary_failure.value} | "
            f"Stage: {self.root_cause_stage.value}"
        )
        lines.append(line1)

        # Line 2: Should answer, risk, confidence
        line2 = (
            f"Should answer: {self.should_have_answered} | "
            f"Risk: {self.security_risk.value} | "
            f"Confidence: {confidence_str}"
        )
        lines.append(line2)

        # Line 3: NCV pipeline health (if present)
        if self.pipeline_health_score is not None or self.first_failing_node is not None:
            health_text = (
                f"{self.pipeline_health_score:.0%}"
                if self.pipeline_health_score is not None
                else "N/A"
            )
            first_failure = self.first_failing_node if self.first_failing_node is not None else "None"
            lines.append(f"Pipeline health: {health_text} | First failure: {first_failure}")

        # Line 4: Failure chain (if present)
        if self.failure_chain:
            chain_str = " → ".join(self.failure_chain)
            lines.append(f"Failure chain: {chain_str}")

        # Line 5: Semantic entropy (if present)
        if self.semantic_entropy is not None:
            lines.append(f"Semantic entropy: {self.semantic_entropy:.2f}")

        # Line 6: Root cause (if present)
        if self.root_cause_attribution:
            lines.append(f"Root cause: {self.root_cause_attribution}")

        # Line 7: Fix (proposed fix takes precedence over recommended fix)
        fix_text = self.proposed_fix if self.proposed_fix else self.recommended_fix
        if self.fix_confidence is not None:
            lines.append(f"Fix ({self.fix_confidence:.0%} confidence): {fix_text}")
        else:
            lines.append(f"Fix: {fix_text}")

        return "\n".join(lines)
