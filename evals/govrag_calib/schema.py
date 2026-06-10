"""
Pydantic schema for the GovRAG Calibration Dataset (GovRAG-Calib).

Each record (GovRAGCalibCase) represents one complete RAG pipeline run
with human-verified expected diagnosis labels, claim-level annotations,
citation annotations, and split/source metadata.

This schema is the ground truth contract for the calibration loop.
No analyzer logic is included here — this is a pure data schema.

Method status: heuristic_baseline (schema only, no statistical calibration yet)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies  (mirror raggov.models.diagnosis enums as strings
# so schema.py has no runtime dependency on the src package)
# ---------------------------------------------------------------------------

FailureTypeStr = Literal[
    "CLEAN",
    "CITATION_MISMATCH",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "STALE_RETRIEVAL",
    "SCOPE_VIOLATION",
    "PROMPT_INJECTION",
    "INSUFFICIENT_CONTEXT",
    "RETRIEVAL_ANOMALY",
    "RETRIEVAL_DEPTH_LIMIT",
    "INCONSISTENT_CHUNKS",
    "POST_RATIONALIZED_CITATION",
    "LOW_CONFIDENCE",
    "PRIVACY_VIOLATION",
    "SUSPICIOUS_CHUNK",
    "TABLE_STRUCTURE_LOSS",
    "HIERARCHY_FLATTENING",
    "METADATA_LOSS",
    "PARSER_STRUCTURE_LOSS",
    "CHUNKING_BOUNDARY_ERROR",
    "EMBEDDING_DRIFT",
    "RERANKER_FAILURE",
    "GENERATION_IGNORE",
    "INCOMPLETE_DIAGNOSIS",
]

FailureStageStr = Literal[
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "RETRIEVAL",
    "RERANKING",
    "CONTEXT_ASSEMBLY",
    "GROUNDING",
    "SUFFICIENCY",
    "GENERATION",
    "CITATION",
    "SECURITY",
    "CONFIDENCE",
    "UNKNOWN",
]

LabelSource = Literal[
    "human",
    "synthetic_mutation",
    "public_dataset_mapped",
    "benchmark_migrated",
]

LabelConfidence = Literal["high", "medium", "low"]

Split = Literal["train", "dev", "heldout", "unset"]

ClaimLabelValue = Literal[
    "supported",
    "unsupported",
    "contradicted",
    "insufficient_evidence",
    "partial",
]

CitationLabelValue = Literal[
    "supports",
    "does_not_support",
    "phantom",
    "post_rationalized",
    "missing_required",
    "contradicted",
]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """Minimal chunk representation portable across evaluation scripts."""

    chunk_id: str = Field(description="Unique ID for this chunk within the case.")
    doc_id: str = Field(description="Source document ID this chunk belongs to.")
    text: str = Field(description="Raw text content of the chunk.")
    rank: int = Field(description="Retrieval rank (1 = top result).")
    score: float | None = Field(
        default=None,
        description="Retrieval similarity score from the vector store.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (title, source, version, …).",
    )


class ClaimLabel(BaseModel):
    """Expected ground-truth label for a single extracted claim."""

    claim_id: str = Field(description="Unique claim identifier within the case.")
    claim_text: str = Field(description="The verbatim extracted claim sentence.")
    expected_label: ClaimLabelValue = Field(
        description="Human-verified claim grounding verdict."
    )
    expected_evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that provide evidence for the expected_label.",
    )
    expected_failure_reason: str | None = Field(
        default=None,
        description="Free-text explanation when label is not 'supported'.",
    )
    critical_entities: list[str] = Field(
        default_factory=list,
        description="Named entities required for exact-match verification.",
    )
    critical_values: list[str] = Field(
        default_factory=list,
        description="Numeric or quoted values required for exact-match verification.",
    )
    critical_dates: list[str] = Field(
        default_factory=list,
        description="Date strings required for exact-match verification.",
    )


class CitationLabel(BaseModel):
    """Expected ground-truth label for a single citation reference in the answer."""

    citation_id: str = Field(description="Unique citation identifier within the case.")
    cited_doc_id: str = Field(description="The document ID cited by the answer.")
    cited_chunk_id: str | None = Field(
        default=None,
        description="Specific chunk ID cited, if chunk-level citation is trackable.",
    )
    expected_label: CitationLabelValue = Field(
        description="Human-verified citation faithfulness verdict."
    )
    expected_reason: str | None = Field(
        default=None,
        description="Explanation for the verdict, especially for non-'supports' labels.",
    )


# ---------------------------------------------------------------------------
# Main calibration case
# ---------------------------------------------------------------------------


class GovRAGCalibCase(BaseModel):
    """
    One labeled calibration case for GovRAG-Calib.

    A calibration case encodes:
    - the full RAG pipeline input (query + retrieved chunks + answer + citations)
    - expected diagnosis at the failure-type, stage, and node level
    - expected claim-level and citation-level labels
    - expected issue flags per dimension (retrieval, sufficiency, version, security, quality)
    - dataset split and labeling provenance metadata

    This model is the contract between dataset producers and the evaluation harness.
    No calibration logic is embedded here.

    Method status: heuristic_baseline (labels are manual or migrated; no statistical calibration).
    """

    # ---- Identity ----------------------------------------------------------
    case_id: str = Field(
        description="Unique identifier for this calibration case, e.g. 'gc-001'."
    )
    domain: str = Field(
        description=(
            "Domain of the RAG case, e.g. 'software', 'healthcare', 'finance', "
            "'government', 'legal', 'general'."
        )
    )
    source_type: str = Field(
        description=(
            "Origin of the case, e.g. 'fixture', 'benchmark_common', "
            "'benchmark_grounding', 'benchmark_domain_agnostic', "
            "'synthetic', 'real_production_anonymized'."
        )
    )

    # ---- Pipeline input ----------------------------------------------------
    query: str = Field(description="The user query that triggered this RAG run.")
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="Ordered list of retrieved chunks (rank 1 = top)."
    )
    answer: str = Field(description="The generated answer to evaluate.")
    citations: list[str] = Field(
        default_factory=list,
        description="Document IDs explicitly cited in the answer.",
    )

    # ---- Expected primary diagnosis ----------------------------------------
    expected_primary_failure: FailureTypeStr = Field(
        description="The primary failure type this case is designed to exercise."
    )
    expected_stage: FailureStageStr = Field(
        description="The pipeline stage where the primary failure originates."
    )
    expected_first_failing_node: str | None = Field(
        default=None,
        description="The first pipeline node where failure manifests (analyzer name or stage name).",
    )
    expected_root_cause: str | None = Field(
        default=None,
        description="Human description of the root cause for non-CLEAN cases.",
    )

    # ---- Expected secondary failures ---------------------------------------
    expected_secondary_failures: list[FailureTypeStr] = Field(
        default_factory=list,
        description="Secondary failure types expected in addition to the primary.",
    )

    # ---- Claim and citation labels -----------------------------------------
    expected_claim_labels: list[ClaimLabel] = Field(
        default_factory=list,
        description="Per-claim ground truth labels.",
    )
    expected_citation_labels: list[CitationLabel] = Field(
        default_factory=list,
        description="Per-citation ground truth labels.",
    )

    # ---- Dimension issue flags (nullable = not applicable / not labeled) ---
    expected_retrieval_issue: str | None = Field(
        default=None,
        description=(
            "Expected retrieval-layer issue if any, e.g. 'coverage_gap', "
            "'depth_limit', 'stale_source', 'scope_drift', 'score_anomaly', 'none'."
        ),
    )
    expected_sufficiency_issue: str | None = Field(
        default=None,
        description=(
            "Expected sufficiency issue if any, e.g. 'missing_required_entity', "
            "'missing_value', 'missing_date', 'none'."
        ),
    )
    expected_version_issue: str | None = Field(
        default=None,
        description=(
            "Expected version/staleness issue if any, e.g. 'outdated_doc', "
            "'version_mismatch', 'deprecated_api', 'none'."
        ),
    )
    expected_answer_quality_issue: str | None = Field(
        default=None,
        description=(
            "Expected answer-quality issue if any, e.g. 'fabricated_specifics', "
            "'ignored_context', 'low_confidence_hedge_missing', 'none'."
        ),
    )
    expected_security_issue: str | None = Field(
        default=None,
        description=(
            "Expected security issue if any, e.g. 'prompt_injection_chunk', "
            "'poisoned_chunk', 'privacy_pii', 'none'."
        ),
    )

    # ---- Fix metadata ------------------------------------------------------
    expected_fix_category: str | None = Field(
        default=None,
        description=(
            "High-level fix category, e.g. 'expand_retrieval', 'reindex_docs', "
            "'add_citation_grounding', 'sanitize_corpus', 'abstain'."
        ),
    )
    expected_human_review_required: bool = Field(
        default=False,
        description="Whether a human reviewer should inspect this case before action.",
    )

    # ---- Label provenance --------------------------------------------------
    label_source: LabelSource = Field(
        description="How the labels were produced."
    )
    label_confidence: LabelConfidence = Field(
        description="Annotator confidence in the labels."
    )

    # ---- Dataset split -----------------------------------------------------
    split: Split = Field(
        default="unset",
        description="Dataset split assignment.",
    )

    # ---- Notes -------------------------------------------------------------
    notes: str | None = Field(
        default=None,
        description="Free-text annotation, edge-case notes, or TODO markers.",
    )

    # ---- Validators --------------------------------------------------------

    @field_validator("retrieved_chunks")
    @classmethod
    def chunks_must_have_unique_ids(cls, v: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ids = [c.chunk_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved_chunks must have unique chunk_id values.")
        return v

    @model_validator(mode="after")
    def claim_evidence_ids_must_reference_chunks(self) -> "GovRAGCalibCase":
        chunk_ids = {c.chunk_id for c in self.retrieved_chunks}
        for cl in self.expected_claim_labels:
            for eid in cl.expected_evidence_chunk_ids:
                if eid not in chunk_ids:
                    raise ValueError(
                        f"ClaimLabel '{cl.claim_id}' references chunk_id '{eid}' "
                        f"which is not in retrieved_chunks."
                    )
        return self

    @model_validator(mode="after")
    def citation_ids_must_reference_docs_or_chunks(self) -> "GovRAGCalibCase":
        chunk_ids = {c.chunk_id for c in self.retrieved_chunks}
        doc_ids = {c.doc_id for c in self.retrieved_chunks}
        for cit in self.expected_citation_labels:
            if cit.cited_doc_id not in doc_ids and cit.cited_doc_id not in self.citations:
                # Allow phantom citations that reference doc IDs not in retrieved_chunks
                # (that is the definition of CITATION_MISMATCH / phantom).
                # Validate only that chunk_id references are valid when present.
                pass
            if cit.cited_chunk_id is not None and cit.cited_chunk_id not in chunk_ids:
                raise ValueError(
                    f"CitationLabel '{cit.citation_id}' references cited_chunk_id "
                    f"'{cit.cited_chunk_id}' which is not in retrieved_chunks."
                )
        return self
