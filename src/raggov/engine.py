"""Core orchestration engine for running RagGov analyses."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.confidence.confidence import ConfidenceAnalyzer
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.analyzers.security.anomalies import RetrievalAnomalyAnalyzer
from raggov.analyzers.security.injection import PromptInjectionAnalyzer
from raggov.analyzers.security.poisoning import PoisoningHeuristicAnalyzer
from raggov.analyzers.security.privacy import PrivacyAnalyzer
from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.analyzers.taxonomy_classifier.layer6 import Layer6TaxonomyClassifier
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
            # Pass prior results to ConfidenceAnalyzer, A2PAttributionAnalyzer, Layer6TaxonomyClassifier, and SemanticEntropyAnalyzer
            if isinstance(
                analyzer,
                (ConfidenceAnalyzer, A2PAttributionAnalyzer, Layer6TaxonomyClassifier, SemanticEntropyAnalyzer),
            ):
                analyzer.config = {
                    **self.config,
                    **analyzer.config,
                    "prior_results": list(results),
                }
            results.append(self._run_analyzer(analyzer, run))

        primary_failure = self._primary_failure(results)
        primary_result = self._primary_result(results, primary_failure)
        secondary_failures = self._secondary_failures(results, primary_failure)

        # Check if A2P ran and produced an attribution
        a2p_result = next(
            (r for r in results if r.analyzer_name == "A2PAttributionAnalyzer" and r.status == "fail"),
            None,
        )

        # Use A2P's stage if available, otherwise use primary_result's stage
        if a2p_result is not None and a2p_result.attribution_stage is not None:
            root_cause_stage = a2p_result.attribution_stage
        elif primary_result is not None and primary_result.stage is not None:
            root_cause_stage = primary_result.stage
        else:
            root_cause_stage = FailureStage.UNKNOWN

        confidence = self._confidence(results)

        # Extract A2P attribution fields if present
        root_cause_attribution = None
        proposed_fix = None
        fix_confidence = None
        if a2p_result is not None:
            if a2p_result.evidence:
                root_cause_attribution = a2p_result.evidence[0]  # Abduction reasoning
            proposed_fix = a2p_result.proposed_fix
            fix_confidence = a2p_result.fix_confidence

        # Extract Layer6 report if present
        layer6_result = next(
            (r for r in results if r.analyzer_name == "Layer6TaxonomyClassifier"),
            None,
        )
        layer6_report = None
        failure_chain = []
        if layer6_result is not None and layer6_result.evidence:
            try:
                layer6_report = json.loads(layer6_result.evidence[0])
                failure_chain = layer6_report.get("failure_chain", [])
            except (json.JSONDecodeError, IndexError):
                logger.warning("Failed to parse Layer6 report from evidence")

        # Extract semantic entropy if present
        semantic_entropy_result = next(
            (r for r in results if r.analyzer_name == "SemanticEntropyAnalyzer"),
            None,
        )
        semantic_entropy = None
        if semantic_entropy_result is not None and semantic_entropy_result.score is not None:
            semantic_entropy = semantic_entropy_result.score

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
            root_cause_attribution=root_cause_attribution,
            proposed_fix=proposed_fix,
            fix_confidence=fix_confidence,
            layer6_report=layer6_report,
            failure_chain=failure_chain,
            semantic_entropy=semantic_entropy,
        )

    def _default_analyzers(self) -> list[BaseAnalyzer]:
        """Return default analyzer suite in optimal execution order.

        Order matters:
        1. Retrieval analyzers (independent)
        2. Sufficiency analyzer (independent)
        3. Grounding analyzer (independent)
        4. Security analyzers (independent)
        5. Layer6TaxonomyClassifier (needs prior results from 1-4)
        6. ConfidenceAnalyzer (needs prior results)
        7. SemanticEntropyAnalyzer (optional, needs llm_fn and prior results)
        8. A2PAttributionAnalyzer (runs last, needs all prior results)
        """
        analyzers = [
            # Retrieval stage analyzers
            ScopeViolationAnalyzer(self.config),
            StaleRetrievalAnalyzer(self.config),
            CitationMismatchAnalyzer(self.config),
            InconsistentChunksAnalyzer(self.config),
            # Sufficiency stage
            SufficiencyAnalyzer(self.config),
            # Grounding stage
            ClaimGroundingAnalyzer(self.config),
            # Security stage analyzers
            PromptInjectionAnalyzer(self.config),
            RetrievalAnomalyAnalyzer(self.config),
            PoisoningHeuristicAnalyzer(self.config),
            PrivacyAnalyzer(self.config),
            # Taxonomy classification (needs prior results)
            Layer6TaxonomyClassifier(self.config),
            # Confidence analyzers (need prior results)
            ConfidenceAnalyzer(self.config),
        ]

        # Add SemanticEntropyAnalyzer if LLM is enabled
        if self.config.get("use_llm", False) and self.config.get("llm_fn") is not None:
            analyzers.append(SemanticEntropyAnalyzer(self.config))

        # Add A2P attribution analyzer last if enabled (needs all prior results)
        if self.config.get("enable_a2p", False):
            analyzers.append(A2PAttributionAnalyzer(self.config))

        return analyzers

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
