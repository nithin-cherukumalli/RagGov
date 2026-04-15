"""Audit log input and output utilities for RagGov."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raggov.io.serialize import diagnosis_to_dict
from raggov.models.diagnosis import Diagnosis
from raggov.models.run import RAGRun


class AuditLog:
    """Append-only JSONL audit log for RagGov diagnoses."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, run: RAGRun, diagnosis: Diagnosis) -> None:
        """Append one diagnosis summary entry to the audit log."""
        diagnosis_data = diagnosis_to_dict(diagnosis)
        entry = {
            "run_id": run.run_id,
            "query": run.query,
            "primary_failure": diagnosis_data["primary_failure"],
            "security_risk": diagnosis_data["security_risk"],
            "should_have_answered": diagnosis.should_have_answered,
            "confidence": diagnosis.confidence,
            "created_at": diagnosis_data["created_at"],
            "checks_run": diagnosis.checks_run,
        }
        with self.path.open("a") as file:
            file.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read all audit log entries."""
        with self.path.open() as file:
            return [json.loads(line) for line in file if line.strip()]

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last n audit log entries."""
        return self.read_all()[-n:]
