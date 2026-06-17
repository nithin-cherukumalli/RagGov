"""Serialization utilities for RagGov models and analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from raggov.models.diagnosis import Diagnosis, FailureType
from raggov.models.run import RAGRun


def diagnosis_why_block(diagnosis: Diagnosis) -> dict[str, Any]:
    """Derive the engineer-facing provenance summary from a Diagnosis.

    This is pure serialization glue. It does not add model fields or change
    analyzer decisions.
    """
    voters = _voting_analyzers(diagnosis)
    also_considered = _also_considered(diagnosis)
    return {
        "verdict_summary": _verdict_summary(diagnosis, voters),
        "voted_by": voters,
        "also_considered": also_considered,
        "inspect_next": _inspect_next_target(diagnosis),
    }


def diagnosis_to_dict(diagnosis: Diagnosis) -> dict[str, Any]:
    """Serialize a diagnosis to a plain JSON-compatible dictionary."""
    payload = diagnosis.model_dump(mode="json")
    payload["why_block"] = diagnosis_why_block(diagnosis)
    return payload


def diagnosis_to_json(diagnosis: Diagnosis, indent: int = 2) -> str:
    """Serialize a diagnosis to a JSON string."""
    return json.dumps(diagnosis_to_dict(diagnosis), indent=indent)


def run_to_dict(run: RAGRun) -> dict[str, Any]:
    """Serialize a RAG run to a plain JSON-compatible dictionary."""
    return run.model_dump(mode="json")


def load_run(path: str | Path) -> RAGRun:
    """Load and validate a RAGRun from a JSON file."""
    try:
        with Path(path).open() as file:
            data = json.load(file)
        return RAGRun.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid RAGRun JSON in {path}: {exc}") from exc


def _verdict_summary(diagnosis: Diagnosis, voters: list[str]) -> str:
    if diagnosis.primary_failure == FailureType.CLEAN:
        warn_count = sum(1 for result in diagnosis.analyzer_results if result.status == "warn")
        if warn_count:
            return (
                f"CLEAN: {warn_count} analyzer(s) raised advisory warnings, "
                "but none escalated to fail."
            )
        return "CLEAN: no analyzer flagged a failure."
    if voters:
        return f"{diagnosis.primary_failure.value}: selected from {', '.join(voters)}."
    return f"{diagnosis.primary_failure.value}: selected by engine-level decision."


def _voting_analyzers(diagnosis: Diagnosis) -> list[str]:
    if diagnosis.primary_failure == FailureType.CLEAN:
        return []
    return [
        result.analyzer_name
        for result in diagnosis.analyzer_results
        if result.status == "fail" and result.failure_type == diagnosis.primary_failure
    ]


def _also_considered(diagnosis: Diagnosis) -> list[dict[str, str]]:
    primary = diagnosis.primary_failure
    seen: set[tuple[str, str, str]] = set()
    considered: list[dict[str, str]] = []
    for result in diagnosis.analyzer_results:
        if result.status not in {"fail", "warn"}:
            continue
        if result.failure_type is None or result.failure_type == primary:
            continue
        key = (result.analyzer_name, result.failure_type.value, result.status)
        if key in seen:
            continue
        seen.add(key)
        considered.append(
            {
                "analyzer": result.analyzer_name,
                "failure_type": result.failure_type.value,
                "status": result.status,
            }
        )
    return considered


def _inspect_next_target(diagnosis: Diagnosis) -> dict[str, str] | None:
    if diagnosis.pinpoint_findings:
        loc = diagnosis.pinpoint_findings[0].location
        target: dict[str, str] = {}
        if getattr(loc, "claim_id", None):
            target["claim_id"] = str(loc.claim_id)
        if getattr(loc, "chunk_id", None):
            target["chunk_id"] = str(loc.chunk_id)
        if getattr(loc, "doc_id", None):
            target["doc_id"] = str(loc.doc_id)
        if getattr(loc, "stage", None):
            target["stage"] = str(loc.stage)
        if target:
            target["source"] = "pinpoint_findings"
            return target
    if diagnosis.root_cause_attribution:
        return {"source": "root_cause_attribution", "value": diagnosis.root_cause_attribution}
    if diagnosis.first_failing_node:
        return {"source": "first_failing_node", "node": diagnosis.first_failing_node}
    return None
