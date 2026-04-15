"""Public Python SDK for RagGov."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raggov.engine import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun


__version__ = "0.1.0"

__all__ = [
    "CorpusEntry",
    "Diagnosis",
    "RAGRun",
    "RetrievedChunk",
    "diagnose",
    "diagnose_dict",
    "diagnose_file",
]


def diagnose_dict(data: dict[str, Any], config: dict[str, Any] | None = None) -> Diagnosis:
    """Validate a run dictionary and return its diagnosis."""
    return diagnose(RAGRun.model_validate(data), config=config)


def diagnose_file(path: str | Path, config: dict[str, Any] | None = None) -> Diagnosis:
    """Load a run JSON file and return its diagnosis."""
    with Path(path).open() as file:
        return diagnose_dict(json.load(file), config=config)
