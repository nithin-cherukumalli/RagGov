"""Analyzer for estimating confidence in RagGov diagnoses."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.models.signals import EvidenceSignalMetadata
from raggov.models.findings import AnalyzerReport


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

        sig = EvidenceSignalMetadata(
            signal_name="overall_confidence_score",
            source_analyzer=self.name(),
            method="aggregate_confidence_heuristics",
            method_status="heuristic_baseline",
            calibration_status="uncalibrated",
            evidence_strength="advisory",
            evidence_tier="proxy",
        )

        status = "pass"
        failure_type = None
        remediation = None
        if final_score < low_threshold:
            status = "fail"
            failure_type = FailureType.LOW_CONFIDENCE
            remediation = REMEDIATION
        elif final_score < warn_threshold:
            status = "warn"
            failure_type = FailureType.LOW_CONFIDENCE

        report = AnalyzerReport(
            analyzer_name=self.name(),
            overall_status=status,
            findings=[],
            notes=["This analyzer uses heuristics and does not emit calibrated confidence."]
        )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status=status,
            failure_type=failure_type,
            stage=FailureStage.CONFIDENCE if status != "pass" else None,
            score=final_score,
            evidence=evidence,
            remediation=remediation,
            signal_metadata=[sig],
            analyzer_report=report,
        )

    def _status_penalty(self, status: str) -> float:
        if status == "fail":
            return 0.2
        if status == "warn":
            return 0.1
        if status == "skip":
            return 0.05
        return 0.0
