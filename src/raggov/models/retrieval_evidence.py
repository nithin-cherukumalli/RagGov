"""
Evidence substrate for retrieval-layer analysis in GovRAG.

This module defines shared data structures that retrieval analyzers populate
to describe how retrieved chunks relate to queries, claims, and citations.

IMPORTANT:
- This is an evidence substrate, NOT a final scoring or gating system.
- v0 outputs are heuristic and uncalibrated; they must not gate production flows.
- All method types default to heuristic_baseline until calibrated against gold data.
- Verification labels and scores from this substrate require downstream calibration
  before use in trust or attribution decisions.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class QueryRelevanceLabel(str, Enum):
    """Relevance of a chunk to the query that triggered retrieval."""
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class RelevanceMethod(str, Enum):
    """Method used to determine query relevance for a chunk."""
    LEXICAL_OVERLAP = "lexical_overlap"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    LLM_JUDGE = "llm_judge"
    NLI = "nli"
    UNAVAILABLE = "unavailable"


class CitationStatus(str, Enum):
    """Citation relationship between a chunk and the generated answer."""
    CITED = "cited"
    UNCITED = "uncited"
    PHANTOM = "phantom"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FreshnessStatus(str, Enum):
    """Temporal freshness of the source document backing a chunk."""
    VALID = "valid"
    STALE_BY_AGE = "stale_by_age"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class EvidenceRole(str, Enum):
    """Functional role a chunk plays in supporting or undermining the answer."""
    NECESSARY_SUPPORT = "necessary_support"
    PARTIAL_SUPPORT = "partial_support"
    BACKGROUND = "background"
    NOISE = "noise"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"


class RetrievalMethodType(str, Enum):
    """Epistemological tier of the retrieval analysis method."""
    HEURISTIC_BASELINE = "heuristic_baseline"
    PRACTICAL_APPROXIMATION = "practical_approximation"
    RESEARCH_FAITHFUL = "research_faithful"


class CalibrationStatus(str, Enum):
    """Calibration state of the retrieval evidence profile."""
    UNCALIBRATED = "uncalibrated"
    MOCK_CALIBRATED = "mock_calibrated"
    CALIBRATED = "calibrated"


class ChunkEvidenceProfile(BaseModel):
    """
    Evidence substrate for a single retrieved chunk.

    Populated by retrieval analyzers to describe how a chunk relates to the
    query, claims, and citations. This is NOT a scored output — it is a
    structured intermediate representation for downstream analyzer consumption.

    v0: heuristic_baseline, uncalibrated.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    chunk_id: str
    source_doc_id: Optional[str] = None
    query_relevance_label: QueryRelevanceLabel = QueryRelevanceLabel.UNKNOWN
    query_relevance_score: Optional[float] = None
    relevance_method: RelevanceMethod = RelevanceMethod.UNAVAILABLE
    supported_claim_ids: List[str] = Field(default_factory=list)
    contradicted_claim_ids: List[str] = Field(default_factory=list)
    neutral_claim_ids: List[str] = Field(default_factory=list)
    citation_status: CitationStatus = CitationStatus.UNKNOWN
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    evidence_role: EvidenceRole = EvidenceRole.UNKNOWN
    warnings: List[str] = Field(default_factory=list)


class RetrievalEvidenceProfile(BaseModel):
    """
    Shared evidence substrate aggregating chunk-level profiles for a RAG run.

    This model is the primary data structure for retrieval-layer analysis in
    GovRAG. All retrieval analyzers (CitationMismatch, ScopeViolation,
    InconsistentChunks, StaleRetrieval) can consume and populate this profile.

    IMPORTANT:
    - v0 is heuristic_baseline and uncalibrated.
    - Do not use recommended_for_gating=True until calibrated against gold data.
    - limitations must list all known weaknesses of the current method.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: Optional[str] = None
    overall_retrieval_status: str = "unknown"
    chunks: List[ChunkEvidenceProfile] = Field(default_factory=list)
    missing_evidence_claim_ids: List[str] = Field(default_factory=list)
    noisy_chunk_ids: List[str] = Field(default_factory=list)
    contradictory_pairs: List[Tuple[str, str]] = Field(default_factory=list)
    phantom_citation_doc_ids: List[str] = Field(default_factory=list)
    stale_doc_ids: List[str] = Field(default_factory=list)
    method_type: RetrievalMethodType = RetrievalMethodType.HEURISTIC_BASELINE
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    recommended_for_gating: bool = False
    limitations: List[str] = Field(default_factory=list)
