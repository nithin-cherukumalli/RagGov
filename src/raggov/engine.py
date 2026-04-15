"""Core orchestration engine for running RagGov analyses."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.confidence.confidence import ConfidenceAnalyzer
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.analyzers.security.anomalies import RetrievalAnomalyAnalyzer
from raggov.analyzers.security.injection import PromptInjectionAnalyzer
from raggov.analyzers.security.poisoning import PoisoningHeuristicAnalyzer
from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.diagnosis import (
    AnalyzerResult,
    Diagnosis,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun
from raggov.taxonomy import (
    DEFAULT_REMEDIATIONS,
    FAILURE_PRIORITY,
    should_have_answered,
)


logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """Run analyzers and merge their outputs into a diagnosis."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        analyzers: Sequence[BaseAnalyzer] | None = None,
    ) -> None:
        self.config = config or {}
        self.analyzers = list(analyzers) if analyzers is not None else self._default_analyzers()

    def diagnose(self, run: RAGRun) -> Diagnosis:
        """Run the analyzer suite and return one merged diagnosis."""
        results: list[AnalyzerResult] = []

        for analyzer in self.analyzers:
            if isinstance(analyzer, ConfidenceAnalyzer):
                analyzer.config = {
                    **self.config,
                    **analyzer.config,
                    "prior_results": list(results),
                }
            results.append(self._run_analyzer(analyzer, run))

        primary_failure = self._primary_failure(results)
        primary_result = self._primary_result(results, primary_failure)
        secondary_failures = self._secondary_failures(results, primary_failure)
        root_cause_stage = (
            primary_result.stage
            if primary_result is not None and primary_result.stage is not None
            else FailureStage.UNKNOWN
        )
        confidence = self._confidence(results)

        return Diagnosis(
            run_id=run.run_id,
            primary_failure=primary_failure,
            secondary_failures=secondary_failures,
            root_cause_stage=root_cause_stage,
            should_have_answered=should_have_answered(primary_failure),
            security_risk=_max_risk(results),
            confidence=confidence,
            evidence=self._evidence(results),
            recommended_fix=self._recommended_fix(primary_failure, primary_result),
            checks_run=[result.analyzer_name for result in results],
            checks_skipped=[
                result.analyzer_name for result in results if result.status == "skip"
            ],
            analyzer_results=results,
        )

    def _default_analyzers(self) -> list[BaseAnalyzer]:
        return [
            StaleRetrievalAnalyzer(self.config),
            CitationMismatchAnalyzer(self.config),
            InconsistentChunksAnalyzer(self.config),
            ScopeViolationAnalyzer(self.config),
            SufficiencyAnalyzer(self.config),
            ClaimGroundingAnalyzer(self.config),
            PromptInjectionAnalyzer(self.config),
            RetrievalAnomalyAnalyzer(self.config),
            PoisoningHeuristicAnalyzer(self.config),
            ConfidenceAnalyzer(self.config),
        ]

    def _run_analyzer(self, analyzer: BaseAnalyzer, run: RAGRun) -> AnalyzerResult:
        try:
            return analyzer.analyze(run)
        except Exception as exc:
            logger.exception("Analyzer %s failed", analyzer.name())
            return AnalyzerResult(
                analyzer_name=analyzer.name(),
                status="skip",
                evidence=[str(exc)],
            )

    def _primary_failure(self, results: list[AnalyzerResult]) -> FailureType:
        failed_types = {
            result.failure_type
            for result in results
            if result.status == "fail" and result.failure_type is not None
        }
        for failure_type in FAILURE_PRIORITY:
            if failure_type in failed_types:
                return failure_type
        return FailureType.CLEAN

    def _primary_result(
        self, results: list[AnalyzerResult], primary_failure: FailureType
    ) -> AnalyzerResult | None:
        if primary_failure == FailureType.CLEAN:
            return None
        return next(
            (
                result
                for result in results
                if result.status == "fail" and result.failure_type == primary_failure
            ),
            None,
        )

    def _secondary_failures(
        self, results: list[AnalyzerResult], primary_failure: FailureType
    ) -> list[FailureType]:
        candidate_types = {
            result.failure_type
            for result in results
            if result.status in {"fail", "warn"} and result.failure_type is not None
        }
        return [
            failure_type
            for failure_type in FAILURE_PRIORITY
            if failure_type in candidate_types and failure_type != primary_failure
        ]

    def _evidence(self, results: list[AnalyzerResult]) -> list[str]:
        evidence: list[str] = []
        for result in results:
            if result.status in {"fail", "warn"}:
                evidence.extend(result.evidence)
        return evidence

    def _recommended_fix(
        self, primary_failure: FailureType, primary_result: AnalyzerResult | None
    ) -> str:
        if primary_result is not None and primary_result.remediation:
            return primary_result.remediation
        return DEFAULT_REMEDIATIONS[primary_failure]

    def _confidence(self, results: list[AnalyzerResult]) -> float | None:
        for result in results:
            if result.analyzer_name == "ConfidenceAnalyzer":
                return result.score
        return None


def _max_risk(results: list[AnalyzerResult]) -> SecurityRisk:
    risk_order = {
        SecurityRisk.NONE: 0,
        SecurityRisk.LOW: 1,
        SecurityRisk.MEDIUM: 2,
        SecurityRisk.HIGH: 3,
    }
    max_risk = SecurityRisk.NONE
    for result in results:
        if result.security_risk is None:
            continue
        if risk_order[result.security_risk] > risk_order[max_risk]:
            max_risk = result.security_risk
    return max_risk


def diagnose(run: RAGRun, config: dict[str, Any] | None = None) -> Diagnosis:
    """Diagnose a RAG run with the default engine."""
    return DiagnosisEngine(config=config).diagnose(run)
