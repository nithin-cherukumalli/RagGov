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
    "temporal",
    "obligation",
    "prohibition",
    "causal",
    "comparison",
    "entity_attribute",
    "version_validity",
    "other",
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
    domain: str | None = Field(
        default=None,
        description="Domain of the claim, e.g. 'software', 'healthcare', 'finance', 'government'."
    )

    # ---- Context -----------------------------------------------------------
    query: str = Field(
        description="The original user query that produced this answer."
    )
    answer: str = Field(
        description="The full generated answer text."
    )
    expected_claims: list[str] = Field(
        default_factory=list,
        description="Expected extracted claims for answer-level substrate regression cases.",
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
    expected_cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="Expected cited document ids at claim level when evaluating provenance hooks.",
    )

    # ---- Ground-truth labels -----------------------------------------------
    gold_label: GoldLabel = Field(
        description="Human-verified verification label for this claim."
    )
    expected_support_label: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
        "unverifiable",
        "skipped",
    ] | None = Field(
        default=None,
        description="Expected explicit support label for the claim-evidence substrate.",
    )
    gold_supporting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that genuinely support this claim.",
    )
    expected_supporting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Expected supporting chunk ids exposed by the verifier.",
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
    critical_entities: list[str] = Field(
        default_factory=list,
        description="List of entities critical for exact match verification."
    )
    critical_values: list[str] = Field(
        default_factory=list,
        description="List of numbers/values critical for exact match verification."
    )
    critical_dates: list[str] = Field(
        default_factory=list,
        description="List of dates critical for exact match verification."
    )
    is_compound: bool = Field(
        default=False,
        description="Whether the claim is intentionally compound and requires decomposition."
    )
    expected_safety_gate: str | None = Field(
        default=None,
        description="The expected safety gate trigger if the case is designed to fail a specific gate."
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty level of the case (easy, hard)."
    )
    failure_type: str = Field(
        default="none",
        description="Failure type/category of the case (supported, date_conflict, numeric_conflict, paraphrase, etc.)."
    )

    # ---- Failure analysis --------------------------------------------------
    error_type: ErrorType | None = Field(
        default=None,
        description=(
            "Root cause of a failure when gold_label is not 'entailed'. "
            "None for correctly entailed claims."
        ),
    )
    expected_failure_stage: str | None = Field(
        default=None,
        description="Expected high-level failure stage for the case, when specified.",
    )

    # ---- Annotation notes --------------------------------------------------
    notes: str | None = Field(
        default=None,
        description="Free-text annotation explaining the gold label or edge-case nuance.",
    )
