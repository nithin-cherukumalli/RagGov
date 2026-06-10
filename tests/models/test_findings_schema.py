"""Tests for AnalyzerFinding and AnalyzerReport schemas."""

from __future__ import annotations

import sys
from typing import Literal

import pytest

from raggov.models.diagnosis import FailureStage, FailureType
from raggov.models.findings import AnalyzerFinding, AnalyzerReport
from raggov.models.signals import EvidenceSignalMetadata


def test_analyzer_finding_serializes() -> None:
    metadata = EvidenceSignalMetadata(
        signal_name="low_overlap_chunks",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    finding = AnalyzerFinding(
        finding_id="finding_001",
        analyzer_name="ScopeViolationAnalyzer",
        failure_type=FailureType.SCOPE_VIOLATION,
        stage=FailureStage.RETRIEVAL,
        status="fail",
        severity="high",
        evidence_message="All chunks are irrelevant under lexical overlap.",
        signal_metadata=metadata,
        affected_chunk_ids=["chunk_1", "chunk_2"],
        affected_doc_ids=["doc_1"],
        affected_claim_ids=[],
    )

    data = finding.model_dump()
    assert data["finding_id"] == "finding_001"
    assert data["analyzer_name"] == "ScopeViolationAnalyzer"
    assert data["failure_type"] == "SCOPE_VIOLATION"
    assert data["stage"] == "RETRIEVAL"
    assert data["status"] == "fail"
    assert data["severity"] == "high"
    assert data["signal_metadata"]["signal_name"] == "low_overlap_chunks"
    assert data["affected_chunk_ids"] == ["chunk_1", "chunk_2"]
    assert data["affected_doc_ids"] == ["doc_1"]
    assert data["affected_claim_ids"] == []

    # Verify Pydantic validation and round-trip
    parsed = AnalyzerFinding.model_validate(data)
    assert parsed.finding_id == "finding_001"


def test_analyzer_report_serializes() -> None:
    metadata = EvidenceSignalMetadata(
        signal_name="low_overlap_chunks",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    finding = AnalyzerFinding(
        finding_id="finding_001",
        analyzer_name="ScopeViolationAnalyzer",
        failure_type=FailureType.SCOPE_VIOLATION,
        stage=FailureStage.RETRIEVAL,
        status="fail",
        severity="high",
        evidence_message="All chunks are irrelevant under lexical overlap.",
        signal_metadata=metadata,
    )
    report = AnalyzerReport(
        analyzer_name="ScopeViolationAnalyzer",
        overall_status="fail",
        findings=[finding],
        fallback_used=True,
        fallback_heuristics_used=["legacy_lexical_fallback"],
        notes=["Heuristic run due to missing cross encoder relevance provider."],
        elapsed_ms=12.5,
    )

    data = report.model_dump()
    assert data["analyzer_name"] == "ScopeViolationAnalyzer"
    assert data["overall_status"] == "fail"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["finding_id"] == "finding_001"
    assert data["fallback_used"] is True
    assert data["fallback_heuristics_used"] == ["legacy_lexical_fallback"]
    assert data["notes"] == ["Heuristic run due to missing cross encoder relevance provider."]
    assert data["elapsed_ms"] == 12.5

    parsed = AnalyzerReport.model_validate(data)
    assert parsed.analyzer_name == "ScopeViolationAnalyzer"


def test_analyzer_report_defaults_are_safe() -> None:
    report = AnalyzerReport(
        analyzer_name="DummyAnalyzer",
        overall_status="pass",
    )
    assert report.findings == []
    assert report.fallback_used is False
    assert report.fallback_heuristics_used == []
    assert report.notes == []
    assert report.elapsed_ms is None


def test_finding_accepts_none_failure_type_for_pass_or_skip() -> None:
    metadata = EvidenceSignalMetadata(
        signal_name="overlap_ok",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    finding_pass = AnalyzerFinding(
        finding_id="finding_pass",
        analyzer_name="ScopeViolationAnalyzer",
        failure_type=None,
        stage=None,
        status="pass",
        severity="none",
        evidence_message="Overlap OK.",
        signal_metadata=metadata,
    )
    assert finding_pass.failure_type is None
    assert finding_pass.stage is None


def test_finding_references_signal_metadata() -> None:
    metadata = EvidenceSignalMetadata(
        signal_name="low_overlap_chunks",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    finding = AnalyzerFinding(
        finding_id="finding_001",
        analyzer_name="ScopeViolationAnalyzer",
        failure_type=FailureType.SCOPE_VIOLATION,
        stage=FailureStage.RETRIEVAL,
        status="fail",
        severity="high",
        evidence_message="All chunks are irrelevant.",
        signal_metadata=metadata,
    )
    assert finding.signal_metadata.signal_name == "low_overlap_chunks"
    assert finding.signal_metadata.method_status == "heuristic_baseline"


def test_findings_models_do_not_require_runtime_engine_imports() -> None:
    # Statically verify that findings.py does not import raggov.engine or engine
    import ast
    from pathlib import Path
    
    findings_path = Path(__file__).parents[2] / "src" / "raggov" / "models" / "findings.py"
    tree = ast.parse(findings_path.read_text(encoding="utf-8"))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "engine" not in alias.name, f"Importing engine in findings.py is forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert "engine" not in node.module, f"Importing from engine in findings.py is forbidden: {node.module}"
            for alias in node.names:
                assert "engine" not in alias.name, f"Importing engine components in findings.py is forbidden: {alias.name}"
