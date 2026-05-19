"""PR 5E: Distinguish grounding failures from generation/answer-quality failures."""

from __future__ import annotations

import pytest

from raggov.analyzers.base import BaseAnalyzer
from raggov.decision_policy import EvidenceTier, classify_evidence_tier
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        score=0.9,
    )


class StaticAnalyzer(BaseAnalyzer):
    def __init__(self, result: AnalyzerResult) -> None:
        super().__init__()
        self.result = result

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


def test_incomplete_answer_with_good_context_stage_generation() -> None:
    """UNSUPPORTED_CLAIM from ClaimGroundingAnalyzer with sufficient context → GENERATION stage."""
    grounding_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        evidence=["claim 'steps are 1 and 2' unsupported — context mentions steps 1 2 3 4"],
    )
    sufficiency_result = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="pass",
        sufficiency_result=SufficiencyResult(
            sufficient=True,
            sufficiency_label="sufficient",
            method="heuristic_claim_aware_v0",
            calibration_status="uncalibrated",
        ),
    )
    run = RAGRun(
        query="What are the 4 steps?",
        retrieved_chunks=[chunk("c1", "Step 1, 2, 3, 4")],
        final_answer="The steps are 1 and 2.",
    )
    engine = DiagnosisEngine(analyzers=[StaticAnalyzer(grounding_result), StaticAnalyzer(sufficiency_result)])
    diagnosis = engine.diagnose(run)

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GENERATION


def test_grounding_failure_without_sufficient_context_stays_grounding() -> None:
    """UNSUPPORTED_CLAIM without SufficiencyAnalyzer saying sufficient → stays GROUNDING."""
    grounding_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        evidence=["claim unsupported — no relevant context chunks"],
    )
    run = RAGRun(
        query="What is the policy?",
        retrieved_chunks=[chunk("c1", "unrelated text")],
        final_answer="Policy requires annual renewal.",
    )
    engine = DiagnosisEngine(analyzers=[StaticAnalyzer(grounding_result)])
    diagnosis = engine.diagnose(run)

    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
    assert diagnosis.root_cause_stage == FailureStage.GROUNDING


def test_high_entropy_without_stronger_failure_stage_confidence() -> None:
    """SemanticEntropyAnalyzer text-only uncertainty → STRUCTURED_DIAGNOSTIC tier → beats INSUFFICIENT_CONTEXT."""
    entropy_result = AnalyzerResult(
        analyzer_name="SemanticEntropyAnalyzer",
        status="fail",
        failure_type=FailureType.LOW_CONFIDENCE,
        stage=FailureStage.CONFIDENCE,
        evidence=[
            "claim_label_entropy_proxy_v0 text-only uncertainty: answer contains explicit uncertainty alternatives."
        ],
    )
    retrieval_result = AnalyzerResult(
        analyzer_name="RetrievalDiagnosisAnalyzerV0",
        status="fail",
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        stage=FailureStage.RETRIEVAL,
        evidence=["retrieval miss — insufficient signal"],
    )
    run = RAGRun(
        query="What is the future plan?",
        retrieved_chunks=[chunk("c1", "Plans are vague.")],
        final_answer="We will expand to Mars... or maybe the moon.",
    )
    engine = DiagnosisEngine(analyzers=[StaticAnalyzer(entropy_result), StaticAnalyzer(retrieval_result)])
    diagnosis = engine.diagnose(run)

    assert diagnosis.primary_failure == FailureType.LOW_CONFIDENCE
    assert diagnosis.root_cause_stage == FailureStage.CONFIDENCE


def test_semantic_entropy_text_only_uncertainty_is_structured_diagnostic() -> None:
    """text-only uncertainty evidence promotes SemanticEntropy to STRUCTURED_DIAGNOSTIC tier."""
    result = AnalyzerResult(
        analyzer_name="SemanticEntropyAnalyzer",
        status="fail",
        failure_type=FailureType.LOW_CONFIDENCE,
        stage=FailureStage.CONFIDENCE,
        evidence=[
            "claim_label_entropy_proxy_v0 text-only uncertainty: answer contains explicit uncertainty alternatives."
        ],
    )
    assert classify_evidence_tier(result) == EvidenceTier.STRUCTURED_DIAGNOSTIC


def test_low_confidence_does_not_override_contradicted_claim() -> None:
    """LOW_CONFIDENCE must not beat CONTRADICTED_CLAIM — CONTRADICTED_CLAIM has higher specificity."""
    entropy_result = AnalyzerResult(
        analyzer_name="SemanticEntropyAnalyzer",
        status="fail",
        failure_type=FailureType.LOW_CONFIDENCE,
        stage=FailureStage.CONFIDENCE,
        evidence=[
            "claim_label_entropy_proxy_v0 text-only uncertainty: answer contains explicit uncertainty alternatives."
        ],
    )
    grounding_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.CONTRADICTED_CLAIM,
        stage=FailureStage.GROUNDING,
        evidence=["claim contradicts chunk text"],
    )
    run = RAGRun(
        query="Is smoking allowed?",
        retrieved_chunks=[chunk("c1", "Smoking is strictly prohibited.")],
        final_answer="Smoking is allowed in the breakroom, or maybe it's fine outside.",
    )
    engine = DiagnosisEngine(analyzers=[StaticAnalyzer(entropy_result), StaticAnalyzer(grounding_result)])
    diagnosis = engine.diagnose(run)

    assert diagnosis.primary_failure == FailureType.CONTRADICTED_CLAIM
