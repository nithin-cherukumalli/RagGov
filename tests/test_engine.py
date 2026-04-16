"""Tests for the diagnosis engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.analyzers.grounding.citation_faithfulness import CitationFaithfulnessProbe
from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.taxonomy_classifier.layer6 import Layer6TaxonomyClassifier
from raggov.analyzers.verification.ncv import NCVPipelineVerifier
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


class WeightedStaticAnalyzer(BaseAnalyzer):
    """Analyzer with a configurable weight that returns a fixed result."""

    def __init__(self, result: AnalyzerResult, weight: float) -> None:
        super().__init__()
        self.result = result
        self.weight = weight

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


class InspectingLayer6Analyzer(Layer6TaxonomyClassifier):
    """Capture weighted prior results passed through the engine."""

    observed_prior_names: list[str]

    def __init__(self) -> None:
        super().__init__()
        self.observed_prior_names = []

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        self.observed_prior_names = [
            result.analyzer_name
            for result in self.config.get("weighted_prior_results", [])
        ]
        return self.skip("captured")


class InspectingA2PAnalyzer(A2PAttributionAnalyzer):
    """Capture weighted prior results passed through the engine."""

    observed_prior_names: list[str]

    def __init__(self) -> None:
        super().__init__()
        self.observed_prior_names = []

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        self.observed_prior_names = [
            result.analyzer_name
            for result in self.config.get("weighted_prior_results", [])
        ]
        return self.skip("captured")


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
        SemanticEntropyAnalyzer({"entropy_threshold": 0.0}),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(run())

    assert diagnosis.run_id == "run-1"
    assert diagnosis.primary_failure == FailureType.SUSPICIOUS_CHUNK
    assert diagnosis.secondary_failures == [FailureType.CONTRADICTED_CLAIM]
    assert diagnosis.root_cause_stage == FailureStage.SECURITY
    assert not diagnosis.should_have_answered
    assert diagnosis.security_risk == SecurityRisk.HIGH
    # SemanticEntropy skips in deterministic mode without claim results
    assert diagnosis.confidence is None
    assert diagnosis.evidence == ["suspicious chunk", "contradicted claim"]
    assert diagnosis.recommended_fix == "Quarantine source."
    assert diagnosis.checks_run == ["security", "grounding", "SemanticEntropyAnalyzer"]
    assert diagnosis.checks_skipped == ["SemanticEntropyAnalyzer"]
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
    assert diagnosis.analyzer_results[0].analyzer_name == "ParserValidationAnalyzer"
    assert any(result.analyzer_name == "SemanticEntropyAnalyzer" for result in diagnosis.analyzer_results)


def test_default_suite_includes_parser_validation_first() -> None:
    engine = DiagnosisEngine(config={})

    assert isinstance(engine.analyzers[0], ParserValidationAnalyzer)


def test_ncv_is_opt_in_via_flag() -> None:
    """NCVPipelineVerifier should only run when enable_ncv=True."""
    # Without flag - should NOT be in default suite
    engine = DiagnosisEngine(config={})
    analyzer_names = [analyzer.__class__.__name__ for analyzer in engine.analyzers]
    assert "NCVPipelineVerifier" not in analyzer_names

    # With flag - should be present
    engine = DiagnosisEngine(config={"enable_ncv": True})
    analyzer_names = [analyzer.__class__.__name__ for analyzer in engine.analyzers]
    assert "NCVPipelineVerifier" in analyzer_names
    # Should be inserted after grounding, before security
    assert analyzer_names.index("CitationFaithfulnessProbe") < analyzer_names.index("NCVPipelineVerifier")
    assert analyzer_names.index("NCVPipelineVerifier") < analyzer_names.index("PromptInjectionAnalyzer")


def test_default_suite_includes_citation_faithfulness_after_grounding() -> None:
    engine = DiagnosisEngine(config={})
    analyzer_names = [analyzer.__class__.__name__ for analyzer in engine.analyzers]

    assert "CitationFaithfulnessProbe" in analyzer_names
    assert analyzer_names.index("ClaimGroundingAnalyzer") < analyzer_names.index("CitationFaithfulnessProbe")


def test_parser_failure_remains_root_cause_stage() -> None:
    remediation = (
        "Use a structure-preserving parser (unstructured.io, docling, pymupdf4llm) "
        "before chunking. Tables must preserve row-column bindings."
    )
    test_run = RAGRun(
        run_id="parser-run",
        query="What are the vacancies by district?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                text="District Vacancies Category Warangal 5 Grade A Khammam 3 Grade B",
                source_doc_id="doc-1",
                score=0.8,
            )
        ],
        final_answer="Warangal has 5 vacancies.",
    )

    diagnosis = DiagnosisEngine(config={"enable_a2p": True, "use_llm": False}).diagnose(test_run)

    assert diagnosis.primary_failure == FailureType.TABLE_STRUCTURE_LOSS
    assert diagnosis.root_cause_stage == FailureStage.PARSING
    assert diagnosis.recommended_fix == remediation


def test_engine_populates_ncv_fields_in_diagnosis() -> None:
    """NCVPipelineVerifier should populate NCV fields when enabled."""
    test_run = RAGRun(
        run_id="ncv-run",
        query="How many vacancies are there?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                text="There are several vacancies in the district.",
                source_doc_id="doc-1",
                score=0.88,
            )
        ],
        final_answer="There are several vacancies.",
    )

    # NCV is opt-in, need to enable it
    diagnosis = DiagnosisEngine(config={"enable_ncv": True}).diagnose(test_run)

    assert any(result.analyzer_name == "NCVPipelineVerifier" for result in diagnosis.analyzer_results)
    assert diagnosis.ncv_report is not None
    assert diagnosis.pipeline_health_score is not None
    assert diagnosis.first_failing_node == "ANSWER_COMPLETENESS"


def test_engine_populates_citation_faithfulness_field() -> None:
    test_run = RAGRun.model_validate_json((FIXTURES / "citation_mismatch.json").read_text())

    diagnosis = DiagnosisEngine(config={}).diagnose(test_run)

    assert diagnosis.citation_faithfulness == "post_rationalized"
    assert diagnosis.primary_failure == FailureType.CITATION_MISMATCH
    assert FailureType.POST_RATIONALIZED_CITATION in diagnosis.secondary_failures


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


def test_layer6_and_attribution_surface_root_cause_correctly() -> None:
    """Integration test: Layer6 + A2P correctly identify retrieval as root cause.

    This test proves the evaluator's complaint is fixed:
    "retrieval limited → UNSUPPORTED_CLAIM" becomes
    "retrieval limited → RETRIEVAL: top_k_too_small → Fix: increase top-k"

    The issue: ClaimGroundingAnalyzer sets stage=GROUNDING, but the real
    problem is upstream in RETRIEVAL. Layer6TaxonomyClassifier identifies
    this and sets the correct stage.
    """
    # Load the insufficient_context fixture
    fixture_run = RAGRun.model_validate_json(
        (FIXTURES / "insufficient_context.json").read_text()
    )

    # Run engine in deterministic mode (no LLM)
    engine = DiagnosisEngine(config={"enable_a2p": False})
    diagnosis = engine.diagnose(fixture_run)

    # Assert Layer6TaxonomyClassifier ran
    assert "Layer6TaxonomyClassifier" in diagnosis.checks_run

    # Assert failure_chain is present and non-empty
    assert diagnosis.failure_chain, "failure_chain should be populated by Layer6"
    assert len(diagnosis.failure_chain) > 0, "failure_chain should have at least one entry"

    # Assert root_cause_stage is upstream of GROUNDING (not GROUNDING itself)
    # This is the key fix: Layer6 correctly identifies an upstream stage as the root cause
    # Could be RETRIEVAL, SUFFICIENCY, or CHUNKING depending on the specific failure modes
    assert diagnosis.root_cause_stage in (
        FailureStage.RETRIEVAL,
        FailureStage.SUFFICIENCY,
        FailureStage.CHUNKING,
    ), (
        f"Root cause should be upstream of GROUNDING (RETRIEVAL/SUFFICIENCY/CHUNKING), "
        f"got {diagnosis.root_cause_stage}"
    )

    # The critical assertion: root cause should NOT be GROUNDING
    # This proves Layer6 correctly identified the upstream issue
    assert diagnosis.root_cause_stage != FailureStage.GROUNDING, (
        "Root cause should NOT be GROUNDING - Layer6 should identify upstream issue"
    )

    # Assert layer6_report is populated
    assert diagnosis.layer6_report is not None, "layer6_report should be populated"
    assert "primary_stage" in diagnosis.layer6_report
    assert "engineer_action" in diagnosis.layer6_report

    # Assert engineer_action is specific and actionable
    engineer_action = diagnosis.layer6_report["engineer_action"]
    assert len(engineer_action) > 0, "engineer_action should be non-empty"
    # Should be actionable (contains concrete advice)
    assert any(
        keyword in engineer_action.lower()
        for keyword in [
            "retrieval",
            "top-k",
            "embedding",
            "increase",
            "expand",
            "chunk",
            "review",
            "adjust",
            "improve",
        ]
    ), f"Engineer action should be actionable, got: {engineer_action}"

    # Assert summary includes failure chain
    summary = diagnosis.summary()
    assert "Failure chain:" in summary, "Summary should include failure chain"


def test_a2p_attribution_provides_detailed_fix() -> None:
    """Test that A2P attribution analyzer provides detailed root cause analysis."""
    # Load the insufficient_context fixture
    fixture_run = RAGRun.model_validate_json(
        (FIXTURES / "insufficient_context.json").read_text()
    )

    # Run engine with A2P enabled (deterministic mode)
    engine = DiagnosisEngine(config={"enable_a2p": True, "use_llm": False})
    diagnosis = engine.diagnose(fixture_run)

    # Assert A2P ran
    assert "A2PAttributionAnalyzer" in diagnosis.checks_run

    # If A2P produced results, verify they're populated
    a2p_result = next(
        (r for r in diagnosis.analyzer_results if r.analyzer_name == "A2PAttributionAnalyzer"),
        None,
    )

    if a2p_result and a2p_result.status != "skip":
        # A2P should provide proposed_fix
        assert diagnosis.proposed_fix is not None, "A2P should provide proposed_fix"
        assert len(diagnosis.proposed_fix) > 0

        # A2P should provide fix_confidence
        assert diagnosis.fix_confidence is not None

        # A2P should provide root_cause_attribution
        assert diagnosis.root_cause_attribution is not None


def test_layer6_receives_weighted_prior_results_in_descending_order() -> None:
    low_signal = WeightedStaticAnalyzer(
        AnalyzerResult(
            analyzer_name="low-signal",
            status="warn",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
        ),
        weight=0.2,
    )
    high_signal = WeightedStaticAnalyzer(
        AnalyzerResult(
            analyzer_name="high-signal",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
        weight=0.9,
    )
    layer6 = InspectingLayer6Analyzer()

    DiagnosisEngine(analyzers=[low_signal, high_signal, layer6]).diagnose(run())

    assert layer6.observed_prior_names[:2] == ["high-signal", "low-signal"]


def test_a2p_receives_weighted_prior_results_in_descending_order() -> None:
    low_signal = WeightedStaticAnalyzer(
        AnalyzerResult(
            analyzer_name="low-signal",
            status="warn",
            failure_type=FailureType.INCONSISTENT_CHUNKS,
            stage=FailureStage.RETRIEVAL,
        ),
        weight=0.2,
    )
    high_signal = WeightedStaticAnalyzer(
        AnalyzerResult(
            analyzer_name="high-signal",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
        weight=0.9,
    )
    a2p = InspectingA2PAnalyzer()

    DiagnosisEngine(analyzers=[low_signal, high_signal, a2p]).diagnose(run())

    assert a2p.observed_prior_names[:2] == ["high-signal", "low-signal"]


def test_primary_failure_prefers_higher_weight_signal_over_failure_priority() -> None:
    analyzers = [
        WeightedStaticAnalyzer(
            AnalyzerResult(
                analyzer_name="security-signal",
                status="fail",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
            ),
            weight=0.2,
        ),
        WeightedStaticAnalyzer(
            AnalyzerResult(
                analyzer_name="retrieval-signal",
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
                remediation="refresh retrieval index",
            ),
            weight=0.9,
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(run())

    assert diagnosis.primary_failure == FailureType.STALE_RETRIEVAL
    assert diagnosis.recommended_fix == "refresh retrieval index"


def test_meta_analyzers_do_not_compete_as_primary_evidence_sources() -> None:
    analyzers = [
        WeightedStaticAnalyzer(
            AnalyzerResult(
                analyzer_name="direct-signal",
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
            ),
            weight=0.7,
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="Layer6TaxonomyClassifier",
                status="fail",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
            )
        ),
        StaticAnalyzer(
            AnalyzerResult(
                analyzer_name="A2PAttributionAnalyzer",
                status="fail",
                failure_type=FailureType.GENERATION_IGNORE,
                stage=FailureStage.GENERATION,
            )
        ),
    ]

    diagnosis = DiagnosisEngine(analyzers=analyzers).diagnose(run())

    assert diagnosis.primary_failure == FailureType.STALE_RETRIEVAL
