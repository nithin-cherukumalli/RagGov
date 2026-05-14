"""Pydantic models for curated stress scenarios."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def require_expected_claim_label(self) -> "ClaimExpectation":
        if self.expected_claim_label is None and self.expected_label is None:
            raise ValueError("expected_claims entries require expected_claim_label or expected_label")
        return self


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
    expected_claims: list[ClaimExpectation] = Field(min_length=1)

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

    @model_validator(mode="after")
    def require_expected_sufficiency(self) -> "ClaimDiagnosisGoldCase":
        if self.expected_sufficient is None and self.expected_sufficiency is None:
            raise ValueError(
                "claim diagnosis cases require expected_sufficient or expected_sufficiency"
            )
        return self


class ClaimDiagnosisGoldSet(BaseModel):
    """A set of gold cases for claim-level diagnosis."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(validation_alias=AliasChoices("version", "evaluation_status"))
    cases: list[ClaimDiagnosisGoldCase] = Field(
        default_factory=list,
        validation_alias=AliasChoices("cases", "examples"),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_raw_case_contracts(cls, data: Any) -> Any:
        """Fail with case_id context before nested validation loses that signal."""
        if not isinstance(data, dict):
            return data

        raw_cases = data.get("cases", data.get("examples", []))
        if raw_cases is None:
            return data
        if not isinstance(raw_cases, list):
            raise ValueError("claim diagnosis gold set requires cases/examples to be a list")

        required_case_fields = {
            "case_id",
            "query",
            "final_answer",
            "expected_claims",
            "expected_fix_category",
        }
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise ValueError(f"claim diagnosis case at index {index} must be an object")

            case_id = str(raw_case.get("case_id", f"<index:{index}>"))
            missing = sorted(field for field in required_case_fields if field not in raw_case)
            if "expected_sufficient" not in raw_case and "expected_sufficiency" not in raw_case:
                missing.append("expected_sufficient|expected_sufficiency")
            if "expected_primary_stage" not in raw_case and "expected_stage" not in raw_case:
                missing.append("expected_primary_stage|expected_stage")
            if missing:
                raise ValueError(
                    f"claim diagnosis case '{case_id}' missing required field(s): {', '.join(missing)}"
                )

            expected_claims = raw_case.get("expected_claims")
            if not isinstance(expected_claims, list) or not expected_claims:
                raise ValueError(
                    f"claim diagnosis case '{case_id}' requires non-empty expected_claims"
                )
            for claim_index, claim in enumerate(expected_claims):
                if not isinstance(claim, dict):
                    raise ValueError(
                        f"claim diagnosis case '{case_id}' expected_claims[{claim_index}] must be an object"
                    )
                claim_missing = []
                if "claim_text" not in claim:
                    claim_missing.append("claim_text")
                if "expected_claim_label" not in claim and "expected_label" not in claim:
                    claim_missing.append("expected_claim_label|expected_label")
                if claim_missing:
                    raise ValueError(
                        f"claim diagnosis case '{case_id}' expected_claims[{claim_index}] "
                        f"missing required field(s): {', '.join(claim_missing)}"
                    )
        return data

    @property
    def evaluation_status(self) -> str:
        """Legacy fixture status name retained for the claim diagnosis harness."""
        return self.version

    @property
    def examples(self) -> list[ClaimDiagnosisGoldCase]:
        """Legacy case list name retained for existing harness/report code."""
        return self.cases


class RAGFailureGoldenCase(BaseModel):
    """A comprehensive golden case for RAG pipeline failures."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    description: str | None = None
    query: str
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    cited_doc_ids: list[str] = Field(default_factory=list)

    # Expected diagnosis fields
    expected_primary_failure: str
    expected_root_cause_stage: str
    expected_first_failing_node: str | None = None
    expected_pinpoint_node: str | None = None
    expected_recommended_fix_category: str
    expected_secondary_failures: list[str] = Field(default_factory=list)
    expected_human_review_required: bool
    expected_should_have_answered: bool

    # External signal expectations for the bridge
    expected_external_signals: list[dict[str, Any]] = Field(default_factory=list)
    
    metadata: dict[str, Any] = Field(default_factory=dict)
