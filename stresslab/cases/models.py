"""Pydantic models for curated stress scenarios."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
