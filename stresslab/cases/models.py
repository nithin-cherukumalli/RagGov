"""Pydantic models for curated stress scenarios."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StressCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    document_set: list[str] = Field(default_factory=list)
    query: str
    gold_answer: str | None = None
    gold_supporting_locations: list[str] = Field(default_factory=list)
    pipeline_variant: str
    failure_injection: str
    expected_primary_failure: str
    expected_secondary_failures: list[str] = Field(default_factory=list)
    expected_should_have_answered: bool
    severity: str
    deterministic_or_query_dependent: str


class DiagnosisGoldenCase(BaseModel):
    """Expected diagnosis for a stable fixture-backed RagGov run."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_fixture: str
    expected_primary_failure: str
    expected_root_cause_stage: str
    expected_should_have_answered: bool
    expected_secondary_failures: list[str] = Field(default_factory=list)
    expected_citation_faithfulness: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ClaimExpectation(BaseModel):
    """Expected claim-level outputs for a single extracted claim.

    Evaluation axes:
    - Axis A (Claim Support): expected_claim_label (entailed/unsupported/contradicted)
    - Axis B (Citation Validity): expected_citation_validity (valid/invalid/not_applicable)
    - Axis C (Freshness Validity): expected_freshness_validity (fresh/stale/unknown)
    - Axis E (A2P Root Cause): expected_a2p_primary_cause
    """

    model_config = ConfigDict(extra="forbid")

    claim_text: str

    # Axis A: Claim Support - Does retrieved text support the claim?
    expected_claim_label: str | None = None

    # Backward compatibility
    expected_label: str | None = None

    # Axis B: Citation Validity - Are citations correct and faithful?
    expected_citation_validity: str | None = None

    # Axis C: Freshness Validity - Is the source current?
    expected_freshness_validity: str | None = None

    # Axis E: A2P Root Cause
    expected_a2p_primary_cause: str | None = None


class ClaimDiagnosisGoldCase(BaseModel):
    """Gold case for claim-level diagnosis evaluation.

    Evaluation axes (case-level):
    - Axis D (Context Sufficiency): expected_sufficient (backward compat) or expected_sufficiency
    - Axis E (A2P Root Cause): tracked per-claim in expected_claims
    - Stage/Fix: expected_primary_stage, expected_fix_category
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str = "uncategorized"
    query: str
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    expected_claims: list[ClaimExpectation] = Field(default_factory=list)

    # Axis D: Context Sufficiency
    expected_sufficient: bool | None = None
    expected_sufficiency: bool | None = None

    # Stage and fix category
    expected_primary_stage: str = Field(
        validation_alias=AliasChoices("expected_primary_stage", "expected_stage"),
        serialization_alias="expected_stage",
    )
    expected_fix_category: str

    cited_doc_ids: list[str] = Field(default_factory=list)
    corpus_entries: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimDiagnosisGoldSet(BaseModel):
    """Collection of claim-level diagnosis gold cases."""

    model_config = ConfigDict(extra="forbid")

    evaluation_status: str
    examples: list[ClaimDiagnosisGoldCase] = Field(default_factory=list)
