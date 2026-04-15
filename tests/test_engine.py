"""Tests for the diagnosis engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.confidence.confidence import ConfidenceAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.cli import app
from raggov.engine import DiagnosisEngine, _max_risk, diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    Diagnosis,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun
from raggov.taxonomy import DEFAULT_REMEDIATIONS


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
runner = CliRunner()


class StaticAnalyzer(BaseAnalyzer):
    """Analyzer that returns a fixed result."""

    def __init__(self, result: AnalyzerResult) -> None:
        super().__init__()
        self.result = result

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


class CrashingAnalyzer(BaseAnalyzer):
    """Analyzer that raises to test engine isolation."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        raise RuntimeError("boom")


class FaultyAnalyzer(BaseAnalyzer):
    """Analyzer that crashes with test crash message."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        raise RuntimeError("test crash")


def run() -> RAGRun:
    return RAGRun(
        run_id="run-1",
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
    )


def test_engine_merges_analyzer_results_into_diagnosis() -> None:
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="security",
                status="fail",
                failure_type=FailureType.SUSPICIOUS_CHUNK,
                stage=FailureStage.SECURITY,
                security_risk=SecurityRisk.HIGH,
                evidence=["suspicious chunk"],
                remediation="Quarantine source.",
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="grounding",
                status="warn",
                failure_type=FailureType.CONTRADICTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=["contradicted claim"],
            )
        ),
        ConfidenceAnalyzer({"low_confidence_threshold": 0.0, "warn_confidence_threshold": 0.0}),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(run())

    assert diagnosis.run_id == "run-1"
    assert diagnosis.primary_failure == FailureType.SUSPICIOUS_CHUNK
    assert diagnosis.secondary_failures == [FailureType.CONTRADICTED_CLAIM]
    assert diagnosis.root_cause_stage == FailureStage.SECURITY
    assert not diagnosis.should_have_answered
    assert diagnosis.security_risk == SecurityRisk.HIGH
    assert diagnosis.confidence == 0.7
    assert diagnosis.evidence == ["suspicious chunk", "contradicted claim"]
    assert diagnosis.recommended_fix == "Quarantine source."
    assert diagnosis.checks_run == ["security", "grounding", "ConfidenceAnalyzer"]
    assert diagnosis.checks_skipped == []
    assert len(diagnosis.analyzer_results) == 3


def test_engine_uses_priority_and_default_remediation() -> None:
    analyzers = [
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="retrieval",
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="security",
                status="fail",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
            )
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(run())

    assert diagnosis.primary_failure == FailureType.PROMPT_INJECTION
    assert diagnosis.secondary_failures == [FailureType.STALE_RETRIEVAL]
    assert diagnosis.root_cause_stage == FailureStage.SECURITY
    assert diagnosis.recommended_fix == DEFAULT_REMEDIATIONS[FailureType.PROMPT_INJECTION]


def test_engine_catches_analyzer_exceptions_and_marks_skip(caplog: pytest.LogCaptureFixture) -> None:
    diagnosis = DiagnosisEngine(analyzers=[CrashingAnalyzer()]).diagnose(run())

    assert diagnosis.primary_failure == FailureType.CLEAN
    assert diagnosis.root_cause_stage == FailureStage.UNKNOWN
    assert diagnosis.should_have_answered
    assert diagnosis.checks_skipped == ["CrashingAnalyzer"]
    assert diagnosis.analyzer_results[0].status == "skip"
    assert diagnosis.analyzer_results[0].evidence == ["boom"]
    assert "Analyzer CrashingAnalyzer failed" in caplog.text


def test_engine_isolates_faulty_analyzer_and_continues_with_others() -> None:
    """Gap 2: One analyzer crash should not stop other analyzers from running."""
    test_run = run()
    test_run.retrieved_chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="context",
            source_doc_id="doc-1",
            score=0.9,
        )
    ]
    test_run.cited_doc_ids = ["doc-phantom"]

    diagnosis = DiagnosisEngine(
        analyzers=[FaultyAnalyzer(), CitationMismatchAnalyzer()]
    ).diagnose(test_run)

    # Assert result is a valid Diagnosis (not an exception)
    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.run_id == "run-1"

    # Assert FaultyAnalyzer appears in checks_skipped
    assert "FaultyAnalyzer" in diagnosis.checks_skipped

    # Assert CitationMismatchAnalyzer ran successfully despite FaultyAnalyzer crash
    assert "CitationMismatchAnalyzer" in diagnosis.checks_run
    assert "CitationMismatchAnalyzer" not in diagnosis.checks_skipped

    # Assert the diagnosis identified the phantom citation from CitationMismatchAnalyzer
    assert diagnosis.primary_failure == FailureType.CITATION_MISMATCH

    # Assert FaultyAnalyzer result has skip status with crash evidence
    faulty_result = next(
        r for r in diagnosis.analyzer_results if r.analyzer_name == "FaultyAnalyzer"
    )
    assert faulty_result.status == "skip"
    assert faulty_result.evidence == ["test crash"]


def test_top_level_diagnose_uses_default_suite() -> None:
    diagnosis = diagnose(run())

    assert diagnosis.run_id == "run-1"
    assert diagnosis.analyzer_results
    assert any(result.analyzer_name == "ConfidenceAnalyzer" for result in diagnosis.analyzer_results)


@pytest.mark.parametrize(
    ("fixture_name", "primary_failure", "should_answer", "security_risk"),
    [
        ("clean_pass.json", FailureType.CLEAN, True, SecurityRisk.NONE),
        ("prompt_injection.json", FailureType.PROMPT_INJECTION, False, SecurityRisk.HIGH),
        ("citation_mismatch.json", FailureType.CITATION_MISMATCH, True, SecurityRisk.NONE),
        ("insufficient_context.json", FailureType.INSUFFICIENT_CONTEXT, False, SecurityRisk.NONE),
        ("poisoned_chunk.json", FailureType.SUSPICIOUS_CHUNK, False, SecurityRisk.HIGH),
        ("stale_retrieval.json", FailureType.STALE_RETRIEVAL, True, SecurityRisk.NONE),
        ("unsupported_claims.json", FailureType.UNSUPPORTED_CLAIM, True, SecurityRisk.NONE),
    ],
)
def test_engine_diagnoses_v1_fixtures(
    fixture_name: str,
    primary_failure: FailureType,
    should_answer: bool,
    security_risk: SecurityRisk,
) -> None:
    fixture_run = RAGRun.model_validate_json((FIXTURES / fixture_name).read_text())

    diagnosis = diagnose(fixture_run)

    assert diagnosis.primary_failure == primary_failure
    assert diagnosis.should_have_answered is should_answer
    assert diagnosis.security_risk == security_risk
    assert diagnosis.analyzer_results


@pytest.mark.parametrize(
    ("fixture_name", "expected_failure"),
    [
        ("clean_pass.json", "CLEAN"),
        ("prompt_injection.json", "PROMPT_INJECTION"),
    ],
)
def test_cli_diagnose_fixture_files(
    tmp_path: Path, fixture_name: str, expected_failure: str
) -> None:
    fixture_path = FIXTURES / fixture_name
    run_id = json.loads(fixture_path.read_text())["run_id"]

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["diagnose", str(fixture_path)])
        output_file = Path(f"{run_id}_diagnosis.json")

        assert result.exit_code == 0
        assert expected_failure in result.output
        assert output_file.exists()
        assert json.loads(output_file.read_text())["primary_failure"] == expected_failure


def test_cli_diagnose_malformed_json_returns_error(tmp_path: Path) -> None:
    """Gap 3: CLI should handle malformed JSON with clear error message."""
    malformed_file = tmp_path / "malformed.json"
    malformed_file.write_text('{"query": "test", "missing_required_fields": true}')

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["diagnose", str(malformed_file)])

        # Assert exit code is 1 (error)
        assert result.exit_code == 1

        # Assert output contains clear error message (not a Python traceback)
        assert "Invalid RAGRun" in result.output or "Error" in result.output
        # Assert no Python traceback
        assert "Traceback" not in result.output


def test_max_risk_orders_security_risk_levels() -> None:
    assert (
        _max_risk(
            [
                AnalyzerResult(
                    analyzer_name="a",
                    status="warn",
                    security_risk=SecurityRisk.LOW,
                ),
                AnalyzerResult(
                    analyzer_name="b",
                    status="fail",
                    security_risk=SecurityRisk.HIGH,
                ),
                AnalyzerResult(analyzer_name="c", status="pass"),
            ]
        )
        == SecurityRisk.HIGH
    )
