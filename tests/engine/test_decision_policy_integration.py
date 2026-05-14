from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


class StaticAnalyzer(BaseAnalyzer):
    def __init__(self, result: AnalyzerResult, weight: float = 1.0) -> None:
        super().__init__()
        self.result = result
        self.weight = weight

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


def _run() -> RAGRun:
    return RAGRun(
        run_id="decision-trace-run",
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


def test_decision_trace_records_selected_and_suppressed_candidates() -> None:
    diagnosis = DiagnosisEngine(
        analyzers=[
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="RetrievalAnomalyAnalyzer",
                    status="fail",
                    failure_type=FailureType.RETRIEVAL_ANOMALY,
                    stage=FailureStage.RETRIEVAL,
                    evidence=["retrieval anomaly"],
                ),
                weight=0.99,
            ),
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="CitationMismatchAnalyzer",
                    status="fail",
                    failure_type=FailureType.CITATION_MISMATCH,
                    stage=FailureStage.RETRIEVAL,
                    evidence=["citation mismatch"],
                ),
                weight=0.10,
            ),
        ]
    ).diagnose(_run())

    assert diagnosis.primary_failure == FailureType.CITATION_MISMATCH
    assert diagnosis.diagnosis_decision_trace is not None
    assert diagnosis.diagnosis_decision_trace["selected_primary_failure"] == "CITATION_MISMATCH"
    assert diagnosis.diagnosis_decision_trace["selected_analyzer"] == "CitationMismatchAnalyzer"
    assert diagnosis.diagnosis_decision_trace["suppressed_candidates"]

def test_retrieval_anomaly_does_not_outrank_citation_mismatch() -> None:
    diagnosis = DiagnosisEngine(
        analyzers=[
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="RetrievalAnomalyAnalyzer",
                    status="fail",
                    failure_type=FailureType.RETRIEVAL_ANOMALY,
                    stage=FailureStage.RETRIEVAL,
                    evidence=["retrieval anomaly"],
                ),
                weight=0.99,
            ),
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="CitationMismatchAnalyzer",
                    status="fail",
                    failure_type=FailureType.CITATION_MISMATCH,
                    stage=FailureStage.RETRIEVAL,
                    evidence=["citation mismatch"],
                ),
                weight=0.10,
            ),
        ]
    ).diagnose(_run())
    assert diagnosis.primary_failure == FailureType.CITATION_MISMATCH

def test_retrieval_anomaly_does_not_outrank_unsupported_claim() -> None:
    diagnosis = DiagnosisEngine(
        analyzers=[
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="RetrievalAnomalyAnalyzer",
                    status="fail",
                    failure_type=FailureType.RETRIEVAL_ANOMALY,
                    stage=FailureStage.RETRIEVAL,
                    evidence=["retrieval anomaly"],
                ),
                weight=0.99,
            ),
            StaticAnalyzer(
                AnalyzerResult(
                    analyzer_name="ClaimGroundingAnalyzer",
                    status="fail",
                    failure_type=FailureType.UNSUPPORTED_CLAIM,
                    stage=FailureStage.GROUNDING,
                    evidence=["unsupported claim"],
                ),
                weight=0.10,
            ),
        ]
    ).diagnose(_run())
    assert diagnosis.primary_failure == FailureType.UNSUPPORTED_CLAIM
