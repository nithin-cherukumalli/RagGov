"""Data models describing analyzer diagnoses and findings."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from raggov.calibration import ConfidenceInterval

from raggov.models.grounding import GroundingEvidenceBundle


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
    label: Literal["entailed", "unsupported", "contradicted", "abstain"]
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    contradicting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    verification_method: str | None = None
    evidence_reason: str | None = None
    calibration_status: Literal["uncalibrated", "calibrated"] | None = None
    fallback_used: bool = False
    value_conflicts: list[dict[str, str]] | None = None
    value_matches: list[dict[str, str]] | None = None


class EvidenceRequirement(BaseModel):
    """One evidence requirement inferred for sufficiency assessment."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    requirement_id: str
    description: str
    requirement_type: Literal[
        "definition",
        "rule",
        "date",
        "authority",
        "scope",
        "exception",
        "procedure",
        "numeric_value",
        "comparison",
        "supersession",
        "citation",
    ]
    importance: Literal["critical", "supporting", "optional"] = "critical"
    query_span: str | None = None
    verifier: Literal["heuristic", "llm_judge", "nli", "human_label"] = "heuristic"


class EvidenceCoverage(BaseModel):
    """Coverage status for one evidence requirement."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    requirement_id: str
    status: Literal[
        "covered",
        "partial",
        "missing",
        "contradicted",
        "stale",
        "unknown",
    ]
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    contradicting_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    verifier: Literal["heuristic", "llm_judge", "nli", "human_label"] = "heuristic"
    confidence: float | None = None


class SufficiencyResult(BaseModel):
    """Structured sufficiency assessment payload."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    sufficient: bool
    sufficiency_label: Literal["sufficient", "insufficient", "partial", "unknown"] = "unknown"
    required_evidence: list[EvidenceRequirement] = Field(default_factory=list)
    coverage: list[EvidenceCoverage] = Field(default_factory=list)
    should_expand_retrieval: bool = False
    should_abstain: bool = False
    threshold_used: float | None = None
    fallback_used: bool = False
    limitations: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    method: str
    calibration_status: Literal[
        "uncalibrated",
        "preliminary_calibrated_v1",
    ] = "uncalibrated"


class CandidateCause(BaseModel):
    """A single candidate root cause hypothesis for a failed/risky claim.

    Represents one possible explanation for why a claim failed, including:
    - Evidence supporting and contradicting this hypothesis
    - Affected claims and chunks
    - Counterfactual intervention (act) and predicted outcome (predict)
    - Transparent heuristic score (uncalibrated)
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    cause_id: str
    cause_type: Literal[
        "insufficient_context_or_retrieval_miss",
        "weak_or_ambiguous_evidence",
        "generation_contradicted_retrieved_evidence",
        "stale_source_usage",
        "citation_mismatch",
        "post_rationalized_citation",
        "verification_uncertainty",
        "adversarial_context",
        "retrieval_noise",
        "unknown",
    ]
    stage: FailureStage
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    affected_chunk_ids: list[str] = Field(default_factory=list)
    supporting_analyzers: list[str] = Field(default_factory=list)
    contradicting_analyzers: list[str] = Field(default_factory=list)
    abduct: str
    act: str
    predict: str
    predicted_fix_effect: Literal[
        "would_likely_fix",
        "would_partially_fix",
        "unlikely_to_fix",
        "unknown",
    ] = "unknown"
    heuristic_score: float | None = None
    score_basis: str | None = None
    calibration_status: Literal["uncalibrated"] = "uncalibrated"


class ClaimAttribution(BaseModel):
    """Claim-level A2P attribution payload (v1 - backward compatible)."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_text: str
    claim_label: str
    candidate_causes: list[str] = Field(default_factory=list)
    primary_cause: str
    abduct: str
    act: str
    predict: str
    evidence: list[str] = Field(default_factory=list)
    affected_chunk_ids: list[str] = Field(default_factory=list)
    attribution_method: str
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    fallback_used: bool = False


class ClaimAttributionV2(BaseModel):
    """Claim-level counterfactual A2P attribution v2.

    Multi-hypothesis attribution with explicit primary/secondary causes,
    candidate scoring, and evidence-based reasoning.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_text: str
    claim_label: str
    primary_cause: str
    secondary_causes: list[str] = Field(default_factory=list)
    candidate_causes: list[CandidateCause] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    recommended_fix: str
    recommended_fix_category: str
    attribution_method: Literal[
        "claim_level_counterfactual_a2p_v2",
        "llm_structured_counterfactual_a2p_v2",
        "legacy_failure_level_heuristic",
    ]
    fallback_used: bool = False
    calibration_status: Literal["uncalibrated"] = "uncalibrated"


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
    analysis_source: Literal[
        "retrieval_evidence_profile", "legacy_heuristic_fallback"
    ] | None = None
    claim_results: list[ClaimResult] | None = None
    claim_attributions: list[ClaimAttribution] | None = None
    claim_attributions_v2: list[ClaimAttributionV2] | None = None
    sufficiency_result: SufficiencyResult | None = None
    remediation: str | None = None
    attribution_stage: FailureStage | None = None
    proposed_fix: str | None = None
    fix_confidence: float | None = None
    citation_probe_results: list[dict[str, Any]] | None = None
    diagnostic_rollup: dict[str, Any] | None = None
    """
    RAGChecker-inspired claim-level diagnostic summary produced by
    ClaimDiagnosticRollupBuilder.  None for analyzers that do not produce
    claim-level evidence records.
    """
    grounding_evidence_bundle: GroundingEvidenceBundle | None = None
    """
    Structured bundle of claim evidence records and diagnostic rollups.
    Used as the primary substrate for downstream taxonomy classification
    and attribution.
    """



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
    confidence_intervals: list[ConfidenceInterval] | None = None

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
