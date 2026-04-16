"""Pydantic models for curated stress scenarios."""

from __future__ import annotations

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
