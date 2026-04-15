"""Serialization utilities for RagGov models and analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun


def diagnosis_to_dict(diagnosis: Diagnosis) -> dict[str, Any]:
    """Serialize a diagnosis to a plain JSON-compatible dictionary."""
    return diagnosis.model_dump(mode="json")


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
