"""Analyzer for estimating confidence in RagGov diagnoses."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


REMEDIATION = (
    "Confidence too low to trust output. Consider abstaining, re-retrieving, "
    "or requesting human review."
)


class ConfidenceAnalyzer(BaseAnalyzer):
    """Aggregate lightweight confidence signals for a RAG run."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        score = 1.0
        evidence = ["base score: 1.00"]

        if run.answer_confidence is not None:
            score = (score + run.answer_confidence) / 2
            evidence.append(
                f"blended caller answer_confidence {run.answer_confidence:.2f}: "
                f"score {score:.2f}"
            )

        for prior_result in self.config.get("prior_results", []):
            penalty = self._status_penalty(prior_result.status)
            if penalty == 0:
                continue
            score -= penalty
            evidence.append(
                f"prior result {prior_result.analyzer_name} status "
                f"{prior_result.status}: -{penalty:.2f}"
            )

        retrieval_scores = [
            chunk.score for chunk in run.retrieved_chunks if chunk.score is not None
        ]
        if retrieval_scores:
            average_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
            if average_retrieval_score < 0.5:
                score -= 0.1
                evidence.append(
                    f"average retrieval score {average_retrieval_score:.2f}: -0.10"
                )
            else:
                evidence.append(
                    f"average retrieval score {average_retrieval_score:.2f}: no penalty"
                )

        final_score = round(min(1.0, max(0.0, score)), 2)
        evidence.append(f"final score: {final_score:.2f}")

        low_threshold = float(self.config.get("low_confidence_threshold", 0.4))
        warn_threshold = float(self.config.get("warn_confidence_threshold", 0.6))

        if final_score < low_threshold:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.LOW_CONFIDENCE,
                stage=FailureStage.CONFIDENCE,
                score=final_score,
                evidence=evidence,
                remediation=REMEDIATION,
            )
        if final_score < warn_threshold:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.LOW_CONFIDENCE,
                stage=FailureStage.CONFIDENCE,
                score=final_score,
                evidence=evidence,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            score=final_score,
            evidence=evidence,
        )

    def _status_penalty(self, status: str) -> float:
        if status == "fail":
            return 0.2
        if status == "warn":
            return 0.1
        if status == "skip":
            return 0.05
        return 0.0
