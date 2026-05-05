"""Node-wise evidence aggregation models for NCV-style pipeline verification.

These models support a practical GovRAG architecture upgrade inspired by
NCV-style verification. They are not a research-faithful NCV implementation,
and they are not RAGChecker, RAGAS, DeepEval, RefChecker, Layer6, or A2P.
Reports remain uncalibrated and are not recommended for production gating.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class NCVNode(str, Enum):
    """Ordered RAG pipeline nodes evaluated by the NCV verifier."""

    QUERY_UNDERSTANDING = "query_understanding"
    PARSER_VALIDITY = "parser_validity"
    RETRIEVAL_COVERAGE = "retrieval_coverage"
    RETRIEVAL_PRECISION = "retrieval_precision"
    CONTEXT_ASSEMBLY = "context_assembly"
    VERSION_VALIDITY = "version_validity"
    CLAIM_SUPPORT = "claim_support"
    CITATION_SUPPORT = "citation_support"
    ANSWER_COMPLETENESS = "answer_completeness"
    SECURITY_RISK = "security_risk"


class NCVNodeStatus(str, Enum):
    """Node-level verifier status."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    UNCERTAIN = "uncertain"
    SKIP = "skip"


class NCVMethodType(str, Enum):
    """Epistemological tier used for an NCV node or report."""

    HEURISTIC_BASELINE = "heuristic_baseline"
    EVIDENCE_AGGREGATION = "evidence_aggregation"
    PRACTICAL_APPROXIMATION = "practical_approximation"


class NCVCalibrationStatus(str, Enum):
    """Calibration state for NCV outputs."""

    UNCALIBRATED = "uncalibrated"
    PRELIMINARY_CALIBRATED = "preliminary_calibrated"
    CALIBRATED = "calibrated"


class NCVEvidenceSignal(BaseModel):
    """One explicit evidence signal consumed by an NCV node."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    signal_name: str
    value: str | float | int | bool | None
    source_report: str | None = None
    source_ids: List[str] = Field(default_factory=list)
    interpretation: str
    limitation: str | None = None


class NCVNodeResult(BaseModel):
    """Structured result for one NCV pipeline node."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    node: NCVNode
    status: NCVNodeStatus
    node_score: float | None = None
    primary_reason: str
    evidence_signals: List[NCVEvidenceSignal] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    affected_claim_ids: List[str] = Field(default_factory=list)
    affected_chunk_ids: List[str] = Field(default_factory=list)
    affected_doc_ids: List[str] = Field(default_factory=list)
    alternative_explanations: List[str] = Field(default_factory=list)
    recommended_fix: str | None = None
    method_type: NCVMethodType
    calibration_status: NCVCalibrationStatus
    fallback_used: bool = False
    limitations: List[str] = Field(default_factory=list)


class NCVReport(BaseModel):
    """Run-level node-wise evidence aggregation report."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    node_results: List[NCVNodeResult]
    first_failing_node: NCVNode | None = None
    first_uncertain_node: NCVNode | None = None
    pipeline_health_score: float | None = None
    bottleneck_description: str
    downstream_failure_chain: List[NCVNode] = Field(default_factory=list)
    evidence_reports_used: List[str] = Field(default_factory=list)
    missing_reports: List[str] = Field(default_factory=list)
    fallback_heuristics_used: List[str] = Field(default_factory=list)
    method_type: NCVMethodType
    calibration_status: NCVCalibrationStatus
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)
