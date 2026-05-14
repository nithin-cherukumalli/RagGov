"""Tests for serialization and audit IO helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from raggov.io.audit import AuditLog
from raggov.io.serialize import (
    diagnosis_to_dict,
    diagnosis_to_json,
    load_run,
    run_to_dict,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import Diagnosis, FailureStage, FailureType, SecurityRisk
from raggov.models.run import RAGRun


def sample_run() -> RAGRun:
    return RAGRun(
        run_id="run-io",
        query="query",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                text="context",
                source_doc_id="doc-1",
                score=0.9,
            )
        ],
        final_answer="answer",
        created_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )


def sample_diagnosis() -> Diagnosis:
    return Diagnosis(
        run_id="run-io",
        primary_failure=FailureType.CLEAN,
        root_cause_stage=FailureStage.UNKNOWN,
        should_have_answered=True,
        security_risk=SecurityRisk.NONE,
        diagnostic_score=0.9,
        ncv_report={"pipeline_health_score": 1.0},
        pipeline_health_score=1.0,
        first_failing_node=None,
        citation_faithfulness="genuine",
        recommended_fix="No remediation is required.",
        checks_run=["SemanticEntropyAnalyzer"],
        created_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
    )


def test_run_and_diagnosis_to_dict_use_iso_datetime_strings() -> None:
    run_dict = run_to_dict(sample_run())
    diagnosis_dict = diagnosis_to_dict(sample_diagnosis())

    assert run_dict["created_at"] == "2026-04-10T12:00:00Z"
    assert diagnosis_dict["created_at"] == "2026-04-10T12:05:00Z"
    assert diagnosis_dict["primary_failure"] == "CLEAN"
    assert diagnosis_dict["security_risk"] == "NONE"
    assert diagnosis_dict["pipeline_health_score"] == 1.0
    assert diagnosis_dict["ncv_report"] == {"pipeline_health_score": 1.0}
    assert diagnosis_dict["citation_faithfulness"] == "genuine"


def test_diagnosis_to_json_returns_json_string() -> None:
    payload = json.loads(diagnosis_to_json(sample_diagnosis(), indent=2))

    assert payload["run_id"] == "run-io"
    assert payload["created_at"] == "2026-04-10T12:05:00Z"


def test_load_run_validates_json_file(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run_to_dict(sample_run())))

    run = load_run(run_file)

    assert run.run_id == "run-io"


def test_load_run_raises_value_error_on_validation_error(tmp_path: Path) -> None:
    run_file = tmp_path / "bad.json"
    run_file.write_text(json.dumps({"run_id": "bad"}))

    with pytest.raises(ValueError, match="Invalid RAGRun JSON"):
        load_run(run_file)


def test_audit_log_appends_reads_and_tails_entries(tmp_path: Path) -> None:
    audit_path = tmp_path / "logs" / "audit.jsonl"
    log = AuditLog(audit_path)

    diagnosis = sample_diagnosis()
    log.append(sample_run(), diagnosis)
    log.append(sample_run(), diagnosis)

    entries = log.read_all()
    assert audit_path.exists()
    assert len(entries) == 2
    assert entries[0] == {
        "run_id": "run-io",
        "query": "query",
        "primary_failure": "CLEAN",
        "security_risk": "NONE",
        "should_have_answered": True,
        "confidence": 0.9,
        "created_at": "2026-04-10T12:05:00Z",
        "checks_run": ["SemanticEntropyAnalyzer"],
    }
    assert log.tail(1) == [entries[-1]]
