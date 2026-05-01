"""
Pydantic schema for the GovRAG claim-grounding gold dataset.

Each record represents a single extracted claim paired with its ground-truth
verification label and supporting evidence metadata.  The schema is designed
to be stable enough to compare heuristic, LLM, and future NLI verifiers
without changing the evaluation harness.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Controlled-vocabulary types
# ---------------------------------------------------------------------------

GoldLabel = Literal["entailed", "unsupported", "contradicted"]

ErrorType = Literal[
    "retrieval_miss",        # The relevant chunk was never retrieved
    "context_ignored",       # The chunk was retrieved but the answer ignored it
    "value_error",           # A numeric/date/identifier value is wrong
    "stale_source_error",    # The answer relies on outdated information
    "citation_error",        # The wrong document was cited for a true claim
    "generation_hallucination",  # The model invented a fact not in any chunk
    "insufficient_context",  # Not enough context to support or refute the claim
]

AtomicityStatus = Literal["atomic", "compound", "unclear"]

ClaimType = Literal[
    "numeric",
    "date_or_deadline",
    "go_number",
    "definition",
    "eligibility",
    "policy_rule",
    "general_factual",
]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """Minimal chunk representation for eval dataset portability."""

    chunk_id: str
    text: str
    source_doc_id: str
    score: float | None = None


# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

class ClaimGroundingCase(BaseModel):
    """
    A single labeled example in the claim-grounding gold dataset.

    Fields mirror the output of ClaimEvidenceBuilder so that metric computation
    can directly compare predicted attributes against gold labels.
    """

    # ---- Identity ----------------------------------------------------------
    case_id: str = Field(
        description="Unique identifier for this case, e.g. 'cgc-001'."
    )

    # ---- Context -----------------------------------------------------------
    query: str = Field(
        description="The original user query that produced this answer."
    )
    answer: str = Field(
        description="The full generated answer text."
    )
    claim_text: str = Field(
        description="The specific extracted claim being evaluated."
    )
    retrieved_chunks: list[ChunkRecord] = Field(
        description="Chunks that were retrieved for this query."
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs explicitly cited by the generated answer.",
    )

    # ---- Ground-truth labels -----------------------------------------------
    gold_label: GoldLabel = Field(
        description="Human-verified verification label for this claim."
    )
    gold_supporting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that genuinely support this claim.",
    )
    gold_contradicting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs whose content contradicts this claim.",
    )

    # ---- Claim metadata ----------------------------------------------------
    claim_type: ClaimType = Field(
        description="Structural type of the claim (numeric, date, policy_rule, …)."
    )
    atomicity_status: AtomicityStatus = Field(
        description="Whether the claim is atomic, compound, or unclear."
    )

    # ---- Failure analysis --------------------------------------------------
    error_type: ErrorType | None = Field(
        default=None,
        description=(
            "Root cause of a failure when gold_label is not 'entailed'. "
            "None for correctly entailed claims."
        ),
    )

    # ---- Annotation notes --------------------------------------------------
    notes: str | None = Field(
        default=None,
        description="Free-text annotation explaining the gold label or edge-case nuance.",
    )
