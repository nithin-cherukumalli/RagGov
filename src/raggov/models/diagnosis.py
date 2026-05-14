"""Data models describing analyzer diagnoses and findings."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from raggov.calibration import ConfidenceInterval
from raggov.evaluators.readiness import ProviderReadiness
from raggov.models.external_diagnosis import ExternalSignalDiagnosisProbe

from raggov.models.citation_faithfulness import CitationFaithfulnessReport
from raggov.models.grounding import GroundingEvidenceBundle
from raggov.models.pinpoint import CausalChain, PinpointFinding, TrustDecision
from raggov.models.retrieval_diagnosis import RetrievalDiagnosisReport
from raggov.models.version_validity import VersionValidityReport


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
    INCOMPLETE_DIAGNOSIS = "INCOMPLETE_DIAGNOSIS"
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
        "required_entity",
        "required_value",
        "required_date_or_time",
        "required_condition_or_scope",
        "required_exception_or_limitation",
        "required_comparison_baseline",
        "required_step_or_procedure",
        "required_causal_support",
        "required_source_or_citation",
        # Legacy aliases accepted for backward compatibility with existing fixtures.
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
    structured_failure_reason: str | None = None
    recommended_fix_category: str | None = None
    evidence_markers: list[str] = Field(default_factory=list)
    missing_requirement_types: list[str] = Field(default_factory=list)
    method: str
    calibration_status: Literal[
        "uncalibrated",
        "preliminary_calibrated_v1",
    ] = "uncalibrated"


class DiagnosisSummary(BaseModel):
    """Actionable structured summary of a GovRAG diagnosis."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    primary_failure: FailureType
    root_cause_stage: FailureStage
    first_failing_node: str | None = None
    pinpoint_location: str | None = None
    selected_evidence: list[str] = Field(default_factory=list)
    suppressed_alternatives: list[FailureType] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    external_provider_state: dict[str, str] = Field(default_factory=dict)
    fallback_heuristics: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    recommended_fix: str
    recommended_next_debug_step: str | None = None


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
    citation_faithfulness_report: CitationFaithfulnessReport | None = None
    version_validity_report: VersionValidityReport | None = None
    retrieval_diagnosis_report: RetrievalDiagnosisReport | None = None
    layer6_report: dict[str, Any] | None = None
    ncv_report: dict[str, Any] | None = None
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
    pinpoint_findings: list[PinpointFinding] = Field(default_factory=list)
    causal_chains: list[CausalChain] = Field(default_factory=list)
    trust_decision: TrustDecision | None = None



class Diagnosis(BaseModel):
    """Overall diagnosis for a RagGov run."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    diagnosis_mode: str = "external-enhanced"
    external_signals_used: list[str] = Field(default_factory=list)
    missing_external_providers: list[str] = Field(default_factory=list)
    external_provider_readiness: list[ProviderReadiness] = Field(default_factory=list)
    fallback_heuristics_used: list[str] = Field(default_factory=list)
    degraded: bool = False
    degraded_external_mode: bool = False  # Alias to degraded for clarity
    external_adapter_errors: list[str] = Field(default_factory=list)
    primary_failure: FailureType
    secondary_failures: list[FailureType] = Field(default_factory=list)
    root_cause_stage: FailureStage
    should_have_answered: bool
    security_risk: SecurityRisk
    claim_results: list[ClaimResult] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommended_fix: str
    checks_run: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)
    analyzer_results: list[AnalyzerResult] = Field(default_factory=list)
    external_diagnosis_probes: list[ExternalSignalDiagnosisProbe] = Field(default_factory=list)
    external_probe_summary: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    root_cause_attribution: str | None = None
    proposed_fix: str | None = None
    fix_confidence: float | None = None
    layer6_report: dict[str, Any] | None = None
    ncv_report: dict[str, Any] | None = None
    pipeline_health_score: float | None = None
    first_failing_node: str | None = None
    first_uncertain_node: str | None = None
    ncv_bottleneck_description: str | None = None
    ncv_downstream_failure_chain: list[str] = Field(default_factory=list)
    ncv_missing_reports: list[str] = Field(default_factory=list)
    ncv_fallback_heuristics_used: list[str] = Field(default_factory=list)
    citation_faithfulness: str | None = None
    citation_faithfulness_report: CitationFaithfulnessReport | None = None
    version_validity_report: VersionValidityReport | None = None
    retrieval_diagnosis_report: RetrievalDiagnosisReport | None = None
    failure_chain: list[str] = Field(default_factory=list)
    semantic_entropy: float | None = None
    heuristic_score: float | None = None
    diagnostic_score: float | None = None
    calibrated_confidence: float | None = None
    calibration_status: str = "uncalibrated"
    confidence_intervals: list[ConfidenceInterval] | None = None
    diagnosis_decision_trace: dict[str, Any] | None = None
    pinpoint_findings: list[PinpointFinding] = Field(default_factory=list)
    causal_chains: list[CausalChain] = Field(default_factory=list)
    trust_decision: TrustDecision | None = None
    summary_v1: DiagnosisSummary | None = None

    @property
    def confidence_label(self) -> str:
        """Return a careful label for the confidence score."""
        if self.calibration_status == "calibrated":
            return "Calibrated Confidence"
        if self.calibration_status == "provisional":
            return "Provisional Confidence"
        return "Confidence Signal (uncalibrated)"

    @property
    def confidence(self) -> float | None:
        """Return the best available confidence measure."""
        if self.calibrated_confidence is not None:
            return self.calibrated_confidence
        # Fallback to diagnostic score if available
        return self.diagnostic_score
    def human_review_required(self) -> bool:
        """Consolidates review requirements across internal trust decisions and external diagnostic probes."""
        # 1. Trigger if explicit trust decision requires it
        if self.trust_decision and self.trust_decision.human_review_required:
            return True
            
        # 2. Trigger if any external diagnostic probe recommends it
        if any(p.should_trigger_human_review for p in self.external_diagnosis_probes):
            return True
            
        # 3. Trigger for specific failure types that are high-risk or represent subtle RAG failures
        # These are cases where the system has detected a specific anomaly that typically
        # requires human validation before making production decisions.
        high_oversight_failures = {
            FailureType.LOW_CONFIDENCE, 
            FailureType.INCOMPLETE_DIAGNOSIS,
            FailureType.UNSUPPORTED_CLAIM,
            FailureType.CONTRADICTED_CLAIM,
            FailureType.CITATION_MISMATCH,
            FailureType.POST_RATIONALIZED_CITATION,
            FailureType.SCOPE_VIOLATION,
            FailureType.INSUFFICIENT_CONTEXT,
            FailureType.RETRIEVAL_ANOMALY,
            FailureType.SUSPICIOUS_CHUNK,
            FailureType.RERANKER_FAILURE,
        }
        if self.primary_failure in high_oversight_failures:
            return True
            
        # 4. Trigger if any secondary failures are high-oversight types
        if any(f in high_oversight_failures for f in self.secondary_failures):
            return True
            
        return False

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the diagnosis."""
        if not self.summary_v1:
            return self._legacy_summary()
        
        s = self.summary_v1
        lines = []
        
        # Actionable Title
        title = f"DIAGNOSIS: {s.primary_failure.value} @ {s.root_cause_stage.value}"
        if s.first_failing_node:
            title += f" (First failing node: {s.first_failing_node})"
        lines.append(title)
        lines.append("=" * len(title))
        
        # Signals
        conf = f"{self.confidence:.2f}" if self.confidence is not None else "N/A"
        lines.append(
            f"Should answer: {self.should_have_answered} | "
            f"Risk: {self.security_risk.value} | "
            f"{self.confidence_label}: {conf} | "
            f"Status: {self.calibration_status}"
        )

        if self.failure_chain:
            chain_str = " → ".join(self.failure_chain)
            lines.append(f"Failure chain: {chain_str}")
        
        if s.human_review_required:
            lines.append("⚠️ HUMAN REVIEW REQUIRED")
            
        # Evidence
        if s.selected_evidence:
            lines.append("\nTop Evidence:")
            for e in s.selected_evidence[:3]:
                lines.append(f"  • {e}")
                
        # Missing/Degraded
        if s.missing_evidence or any(v == "degraded" for v in s.external_provider_state.values()):
            lines.append("\nUncertainty/Degradation:")
            if s.missing_evidence:
                lines.append(f"  • Missing evidence: {', '.join(s.missing_evidence)}")
            degraded = [p for p, st in s.external_provider_state.items() if st == "degraded"]
            if degraded:
                lines.append(f"  • Degraded providers: {', '.join(degraded)}")

        # Next Steps
        lines.append(f"\nRecommended Fix: {s.recommended_fix}")
        if s.recommended_next_debug_step:
            lines.append(f"Next Debug Step: {s.recommended_next_debug_step}")
            
        return "\n".join(lines)

    def _legacy_summary(self) -> str:
        """Old summary implementation for backward compatibility."""
        lines = []

        # Line 1: Run ID, failure type, and stage
        confidence_str = f"{self.confidence:.2f}" if self.confidence is not None else "N/A"
        line1 = (
            f"Run {self.run_id} | {self.primary_failure.value} | "
            f"Stage: {self.root_cause_stage.value}"
        )
        lines.append(line1)
        
        # Incomplete diagnosis details
        if self.primary_failure == FailureType.INCOMPLETE_DIAGNOSIS:
            # Look for critical analyzer or provider names in evidence
            critical_names = {
                "ClaimGroundingAnalyzer", 
                "RetrievalDiagnosisAnalyzerV0", 
                "NCVPipelineVerifier",
                "CitationFaithfulnessAnalyzerV0",
                "ParserValidationAnalyzer"
            }
            missing = [e for e in self.evidence if e in critical_names or e.startswith("External Provider:")]
            if missing:
                lines.append(f"Missing Critical: {', '.join(missing)}")

        # Line 2: Should answer, risk, confidence
        line2 = (
            f"Should answer: {self.should_have_answered} | "
            f"Risk: {self.security_risk.value} | "
            f"{self.confidence_label}: {confidence_str} | "
            f"Status: {self.calibration_status}"
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
            line3 = f"Pipeline health: {health_text} | First failure: {first_failure}"
            if self.first_uncertain_node:
                line3 += f" | First uncertain: {self.first_uncertain_node}"
            lines.append(line3)
            
            if self.ncv_bottleneck_description:
                lines.append(f"Bottleneck: {self.ncv_bottleneck_description}")
            
            if self.ncv_missing_reports:
                lines.append(f"Missing evidence: {', '.join(self.ncv_missing_reports)}")

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
