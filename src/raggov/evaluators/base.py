"""External signal adapter base types and protocols."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from raggov.models.run import RAGRun


class ExternalEvaluatorProvider(str, Enum):
    structured_llm = "structured_llm"
    ragas = "ragas"
    deepeval = "deepeval"
    ragchecker = "ragchecker"
    refchecker = "refchecker"
    cross_encoder = "cross_encoder"
    nli = "nli"
    presidio = "presidio"
    custom = "custom"


class ExternalSignalType(str, Enum):
    claim_support = "claim_support"
    citation_support = "citation_support"
    retrieval_relevance = "retrieval_relevance"
    retrieval_context_precision = "retrieval_context_precision"
    retrieval_context_recall = "retrieval_context_recall"
    retrieval_contextual_relevancy = "retrieval_contextual_relevancy"
    retrieval_contextual_precision = "retrieval_contextual_precision"
    claim_recall = "claim_recall"
    context_utilization = "context_utilization"
    faithfulness = "faithfulness"
    hallucination = "hallucination"
    uncertainty = "uncertainty"
    pii = "pii"
    prompt_injection = "prompt_injection"
    custom = "custom"


class ExternalSignalRecord(BaseModel):
    """One external signal emitted by an adapter."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    provider: ExternalEvaluatorProvider
    signal_type: ExternalSignalType
    metric_name: str
    value: float | str | bool | None = None
    label: str | None = None
    explanation: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    affected_claim_ids: list[str] = Field(default_factory=list)
    affected_chunk_ids: list[str] = Field(default_factory=list)
    affected_doc_ids: list[str] = Field(default_factory=list)
    raw_payload: dict | None = None
    method_type: str = "external_signal_adapter"
    # External signals are never locally calibrated — treat as advisory evidence only.
    calibration_status: str = "uncalibrated_locally"
    # External tools provide signals; GovRAG owns gating decisions.
    recommended_for_gating: bool = False
    limitations: list[str] = Field(default_factory=list)


class ExternalEvaluationResult(BaseModel):
    """Top-level result returned by one external adapter evaluation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    provider: ExternalEvaluatorProvider
    adapter_name: str | None = None
    signals: list[ExternalSignalRecord] = Field(default_factory=list)
    raw_payload: dict | None = None
    succeeded: bool
    error: str | None = None
    missing_dependency: bool = False
    latency_ms: float | None = None
    cost_estimate: float | None = None


@runtime_checkable
class ExternalSignalProvider(Protocol):
    """Minimal protocol every external adapter must satisfy."""

    name: str
    provider: ExternalEvaluatorProvider

    def is_available(self) -> bool: ...
    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult: ...


@runtime_checkable
class ClaimVerifierAdapter(Protocol):
    """Specialized protocol for adapters that verify individual claims."""

    name: str
    provider: ExternalEvaluatorProvider

    def is_available(self) -> bool: ...
    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult: ...
    def verify_claims(
        self, claims: list[str], context: list[str]
    ) -> list[ExternalSignalRecord]: ...


@runtime_checkable
class CitationVerifierAdapter(Protocol):
    """Specialized protocol for adapters that verify citation accuracy."""

    name: str
    provider: ExternalEvaluatorProvider

    def is_available(self) -> bool: ...
    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult: ...
    def verify_citations(
        self, cited_ids: list[str], chunks: list[str]
    ) -> list[ExternalSignalRecord]: ...


@runtime_checkable
class RetrievalSignalProvider(Protocol):
    """Specialized protocol for adapters that score retrieval quality."""

    name: str
    provider: ExternalEvaluatorProvider

    def is_available(self) -> bool: ...
    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult: ...
    def score_relevance(
        self, query: str, chunks: list[str]
    ) -> list[ExternalSignalRecord]: ...
