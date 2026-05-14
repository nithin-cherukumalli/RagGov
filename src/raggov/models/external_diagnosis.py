"""Data models for external signal diagnosis bridging."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExternalSignalDiagnosisProbe(BaseModel):
    """Targeted GovRAG diagnostic probe triggered by an external evaluator signal."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    probe_id: str
    provider: str
    metric_name: str
    signal_type: str
    external_value: float | str | bool | None = None
    external_label: str | None = None
    severity: Literal["low", "medium", "high"]
    suspected_pipeline_node: str
    suspected_failure_stage: str
    suspected_failure_type: str
    affected_claim_ids: list[str] = Field(default_factory=list)
    affected_chunk_ids: list[str] = Field(default_factory=list)
    affected_doc_ids: list[str] = Field(default_factory=list)
    native_analyzers_to_check: list[str] = Field(default_factory=list)
    native_evidence_found: list[str] = Field(default_factory=list)
    native_evidence_missing: list[str] = Field(default_factory=list)
    explanation: str
    recommended_recheck: str
    recommended_fix_category: str
    should_block_clean: bool = False
    should_trigger_human_review: bool = False
    calibration_status: Literal["uncalibrated_locally"] = "uncalibrated_locally"
    recommended_for_gating: bool = False
