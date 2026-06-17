from __future__ import annotations

import pytest

from raggov.analyzers.answer_quality import AnswerQualityAnalyzer
from raggov.analyzers.confidence.confidence import ConfidenceAnalyzer
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, ClaimResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from stresslab.cases.load import load_common_rag_failures
from stresslab.runners.rag_failure_runner import RAGFailureRunner


def chunk(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=score,
    )


def test_answer_incomplete_emits_generation_stage_metadata() -> None:
    run = RAGRun(
        query="What are the 4 steps?",
        retrieved_chunks=[chunk("c1", "Step 1, 2, 3, 4")],
        final_answer="The steps are 1 and 2.",
    )

    result = AnswerQualityAnalyzer().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.UNSUPPORTED_CLAIM
    assert result.stage == FailureStage.GENERATION
    sig = result.signal_metadata[0]
    assert sig.method == "answer_completeness_check"
    assert sig.method_status == "practical_approximation"
    assert sig.calibration_status == "uncalibrated"
    assert sig.evidence_strength == "strong"
    assert result.analyzer_report is not None


def test_answer_ignores_context_emits_generation_stage_metadata() -> None:
    run = RAGRun(
        query="What is the current policy?",
        retrieved_chunks=[chunk("c1", "Policy: No blue shirts.")],
        final_answer="The policy says you can wear anything you want.",
    )
    grounding = ClaimGroundingAnalyzer().analyze(run)

    result = AnswerQualityAnalyzer({"prior_results": [grounding]}).analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.CONTRADICTED_CLAIM
    assert result.stage == FailureStage.GENERATION
    sig = result.signal_metadata[0]
    assert sig.method == "context_adherence_check"
    assert sig.evidence_strength == "strong"
    assert sig.calibration_status == "uncalibrated"


def test_overconfident_weak_evidence_emits_uncalibrated_proxy_metadata() -> None:
    run = RAGRun(
        query="Who won the race?",
        retrieved_chunks=[chunk("c1", "The race was close.", score=0.2)],
        final_answer="Team A definitely won the race.",
    )

    result = AnswerQualityAnalyzer().analyze(run)

    assert result.status == "warn"
    assert result.failure_type == FailureType.LOW_CONFIDENCE
    sig = result.signal_metadata[0]
    assert sig.method == "overconfidence_weak_evidence_check"
    assert sig.evidence_tier == "proxy"
    assert sig.calibration_status == "uncalibrated"


def test_semantic_entropy_sampling_is_uncalibrated_proxy() -> None:
    samples = iter(["A", "B", "C"])

    def llm_fn(_: str) -> str:
        return next(samples, "C")

    run = RAGRun(query="Q?", retrieved_chunks=[chunk("c1", "Context")], final_answer="A")
    result = SemanticEntropyAnalyzer({"use_llm": True, "llm_fn": llm_fn, "n_samples": 3}).analyze(run)

    sig = result.signal_metadata[0]
    assert sig.method == "semantic_entropy_sampling"
    assert sig.calibration_status == "uncalibrated"
    assert sig.evidence_tier == "proxy"


def test_semantic_entropy_proxy_is_weak_or_advisory() -> None:
    prior = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        claim_results=[
            ClaimResult(claim_text="A", label="entailed"),
            ClaimResult(claim_text="B", label="unsupported"),
            ClaimResult(claim_text="C", label="contradicted"),
        ],
    )
    run = RAGRun(
        query="Q?",
        retrieved_chunks=[chunk("c1", "Context")],
        final_answer="A. B. C.",
    )
    result = SemanticEntropyAnalyzer({"use_llm": False, "prior_results": [prior]}).analyze(run)

    sig = result.signal_metadata[0]
    assert sig.method == "semantic_entropy_proxy"
    assert sig.evidence_strength in {"weak", "advisory"}
    assert sig.calibration_status == "uncalibrated"


def test_semantic_entropy_unavailable_is_advisory_unknown() -> None:
    run = RAGRun(query="Q?", retrieved_chunks=[chunk("c1", "Context")], final_answer="A")
    result = SemanticEntropyAnalyzer({"use_llm": True}).analyze(run)

    assert result.status == "skip"
    sig = result.signal_metadata[0]
    assert sig.method == "semantic_entropy_unavailable"
    assert sig.method_status == "external_advisory"
    assert sig.calibration_status == "unknown"
    assert sig.evidence_strength == "advisory"


def test_answer_quality_report_contains_analyzer_findings() -> None:
    run = RAGRun(
        query="What are the 4 steps?",
        retrieved_chunks=[chunk("c1", "Step 1, 2, 3, 4")],
        final_answer="The steps are 1 and 2.",
    )
    result = AnswerQualityAnalyzer().analyze(run)

    assert result.analyzer_report is not None
    assert result.analyzer_report.analyzer_name == "AnswerQualityAnalyzer"
    assert result.analyzer_report.findings
    assert result.analyzer_report.findings[0].stage == FailureStage.GENERATION


@pytest.mark.xfail(
    strict=True,
    reason="Task 15: incomplete-answer case attributes root_cause_stage=GROUNDING "
    "instead of GENERATION. Known routing/stage-attribution bug; see "
    "reports/codex_session/red_test_triage.md and NEXT_TASKS Task 15.",
)
def test_quality_incomplete_38_has_generation_stage_candidate_if_supported() -> None:
    diagnosis = _diagnose_common_case("quality_incomplete_38")

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GENERATION
    trace = diagnosis.diagnosis_decision_trace or {}
    assert trace["selected_analyzer"] == "AnswerQualityAnalyzer"


@pytest.mark.xfail(
    strict=True,
    reason="Task 16: case 41 routes to UNSUPPORTED_CLAIM instead of the more-specific "
    "CONTRADICTED_CLAIM. Known specificity-rank bug; see "
    "reports/codex_session/red_test_triage.md and NEXT_TASKS Task 16.",
)
def test_quality_ignores_context_41_has_generation_stage_candidate_if_supported() -> None:
    diagnosis = _diagnose_common_case("quality_ignores_context_41")

    assert diagnosis.primary_failure == FailureType.CONTRADICTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GENERATION
    trace = diagnosis.diagnosis_decision_trace or {}
    assert trace["selected_analyzer"] == "AnswerQualityAnalyzer"


def test_no_calibrated_confidence_claim_is_made() -> None:
    result = ConfidenceAnalyzer({"warn_confidence_threshold": 1.1}).analyze(
        RAGRun(query="Q?", retrieved_chunks=[chunk("c1", "Context")], final_answer="A")
    )

    assert result.signal_metadata[0].calibration_status == "uncalibrated"
    assert result.analyzer_report is not None
    assert "does not emit calibrated confidence" in result.analyzer_report.notes[0]


def _diagnose_common_case(case_id: str):
    case = next(case for case in load_common_rag_failures() if case.case_id == case_id)
    runner = RAGFailureRunner(mode="native", suite="common")
    run = runner._build_run(case)
    return DiagnosisEngine(config={"mode": "native"}).diagnose(run)
