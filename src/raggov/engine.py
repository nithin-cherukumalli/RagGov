"""Core orchestration engine for running RagGov analyses."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.analyzers.grounding.citation_faithfulness import CitationFaithfulnessProbe
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.analyzers.security.anomalies import RetrievalAnomalyAnalyzer
from raggov.analyzers.security.injection import PromptInjectionAnalyzer
from raggov.analyzers.security.poisoning import PoisoningHeuristicAnalyzer
from raggov.analyzers.security.privacy import PrivacyAnalyzer
from raggov.analyzers.sufficiency.claim_aware import ClaimAwareSufficiencyAnalyzer
from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.analyzers.taxonomy_classifier.layer6 import Layer6TaxonomyClassifier
from raggov.analyzers.verification.ncv import NCVPipelineVerifier
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

    META_ANALYZER_NAMES = {
        "Layer6TaxonomyClassifier",
        "A2PAttributionAnalyzer",
    }
    STATUS_ORDER = {
        "fail": 0,
        "warn": 1,
        "pass": 2,
        "skip": 3,
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        analyzers: Sequence[BaseAnalyzer] | None = None,
    ) -> None:
        self.config = config or {}
        self.analyzers = list(analyzers) if analyzers is not None else self._default_analyzers()

    def diagnose(self, run: RAGRun) -> Diagnosis:
        """Run the research-backed 5-layer analyzer suite and return merged diagnosis.

        Architecture:
        - Layer 1 (Intake Gate): Structural validation
        - Layer 2 (Retrieval Health): Retrieval correctness
        - Layer 3 (Grounding): Answer support verification
        - Layer 4 (Security): System compromise detection
        - Layer 5 (Attribution): Root cause + confidence

        Each layer's results feed into subsequent layers for progressive refinement.
        """
        results: list[AnalyzerResult] = []
        result_weights: dict[str, float] = {}

        for analyzer in self.analyzers:
            # Pass prior results to aggregation analyzers (Layer 5)
            # These analyzers need context from all prior detection layers
            if isinstance(
                analyzer,
                (
                    A2PAttributionAnalyzer,
                    Layer6TaxonomyClassifier,
                    SemanticEntropyAnalyzer,
                    ClaimAwareSufficiencyAnalyzer,
                ),
            ):
                config_update: dict[str, Any] = {
                    **self.config,
                    **analyzer.config,
                    "prior_results": list(results),
                }
                if isinstance(analyzer, (A2PAttributionAnalyzer, Layer6TaxonomyClassifier)):
                    config_update["weighted_prior_results"] = self._weighted_prior_results(
                        results,
                        result_weights,
                    )
                    config_update["analyzer_weights"] = dict(result_weights)
                ncv_report = None
                if isinstance(analyzer, Layer6TaxonomyClassifier):
                    ncv_result = next(
                        (r for r in results if r.analyzer_name == "NCVPipelineVerifier" and r.evidence),
                        None,
                    )
                    if ncv_result is not None:
                        try:
                            ncv_report = json.loads(ncv_result.evidence[0])
                        except (json.JSONDecodeError, IndexError):
                            logger.warning("Failed to parse NCV report from evidence")
                    if ncv_report is not None:
                        config_update["ncv_report"] = ncv_report
                analyzer.config = config_update
            result = self._run_analyzer(analyzer, run)
            results.append(result)
            result_weights.setdefault(result.analyzer_name, analyzer.weight)

        primary_failure = self._primary_failure(results, result_weights)
        primary_result = self._primary_result(results, primary_failure, result_weights)
        secondary_failures = self._secondary_failures(results, primary_failure, result_weights)

        # Check if A2P ran and produced an attribution
        a2p_result = next(
            (r for r in results if r.analyzer_name == "A2PAttributionAnalyzer" and r.status == "fail"),
            None,
        )

        # Only allow A2P to override stage when it carries non-legacy, evidence-backed claim attribution.
        if self._should_use_a2p_stage_override(a2p_result):
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

        # Extract NCV report if present
        ncv_result = next(
            (r for r in results if r.analyzer_name == "NCVPipelineVerifier"),
            None,
        )
        ncv_report = None
        pipeline_health_score = None
        first_failing_node = None
        if ncv_result is not None and ncv_result.evidence:
            try:
                ncv_report = json.loads(ncv_result.evidence[0])
                pipeline_health_score = ncv_report.get("pipeline_health_score")
                first_failing_node = ncv_report.get("first_failing_node")
            except (json.JSONDecodeError, IndexError):
                logger.warning("Failed to parse NCV report from evidence")

        # Extract semantic entropy if present
        semantic_entropy_result = next(
            (r for r in results if r.analyzer_name == "SemanticEntropyAnalyzer"),
            None,
        )
        semantic_entropy = None
        if semantic_entropy_result is not None and semantic_entropy_result.score is not None:
            semantic_entropy = semantic_entropy_result.score

        citation_faithfulness_result = next(
            (r for r in results if r.analyzer_name == "CitationFaithfulnessProbe"),
            None,
        )
        citation_faithfulness = "unchecked"
        if citation_faithfulness_result is not None:
            if citation_faithfulness_result.status == "pass":
                citation_faithfulness = "genuine"
            elif citation_faithfulness_result.status == "fail":
                citation_faithfulness = "post_rationalized"
            elif citation_faithfulness_result.status == "warn":
                citation_faithfulness = "partial"

        grounding_result = next(
            (r for r in results if r.analyzer_name == "ClaimGroundingAnalyzer"),
            None,
        )
        diagnosis_claim_results = grounding_result.claim_results or [] if grounding_result else []

        diagnosis = Diagnosis(
            run_id=run.run_id,
            primary_failure=primary_failure,
            secondary_failures=secondary_failures,
            root_cause_stage=root_cause_stage,
            should_have_answered=should_have_answered(primary_failure),
            security_risk=_max_risk(results),
            confidence=confidence,
            claim_results=diagnosis_claim_results,
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
            ncv_report=ncv_report,
            pipeline_health_score=pipeline_health_score,
            first_failing_node=first_failing_node,
            citation_faithfulness=citation_faithfulness,
            failure_chain=failure_chain,
            semantic_entropy=semantic_entropy,
        )

        if self.config.get("calibrator"):
            diagnosis.confidence_intervals = self.config["calibrator"].calibrate()

        return diagnosis

    def _should_use_a2p_stage_override(self, a2p_result: AnalyzerResult | None) -> bool:
        if a2p_result is None or a2p_result.attribution_stage is None:
            return False

        for attribution in a2p_result.claim_attributions or []:
            if attribution.fallback_used:
                continue
            if attribution.attribution_method == "legacy_failure_level_heuristic":
                continue
            if attribution.evidence:
                return True

        for attribution in a2p_result.claim_attributions_v2 or []:
            if attribution.fallback_used:
                continue
            if attribution.attribution_method == "legacy_failure_level_heuristic":
                continue
            if attribution.evidence_summary:
                return True
            if any(candidate.evidence_for for candidate in attribution.candidate_causes):
                return True

        return False

    def _default_analyzers(self) -> list[BaseAnalyzer]:
        """Return research-backed analyzer suite in optimal 5-layer execution order.

        LAYER 1 — INTAKE GATE: Structural validation before analysis
        LAYER 2 — RETRIEVAL HEALTH: Did retrieval work correctly?
        LAYER 3 — GROUNDING: Is answer supported by context?
        LAYER 4 — SECURITY: Is system compromised?
        LAYER 5 — ATTRIBUTION + CONFIDENCE: Why failed, how confident?

        Each layer feeds the next. Order is critical for proper diagnosis.
        """
        analyzers = [
            # ════════════════════════════════════════════════════════════════
            # LAYER 1 — INTAKE GATE
            # ════════════════════════════════════════════════════════════════
            ParserValidationAnalyzer(self.config),
            SufficiencyAnalyzer(self.config),
            # ════════════════════════════════════════════════════════════════
            # LAYER 2 — RETRIEVAL HEALTH
            # ════════════════════════════════════════════════════════════════
            ScopeViolationAnalyzer(self.config),
            StaleRetrievalAnalyzer(self.config),
            CitationMismatchAnalyzer(self.config),
            # ════════════════════════════════════════════════════════════════
            # LAYER 3 — GROUNDING (chain: grounding → faithfulness probe)
            # ════════════════════════════════════════════════════════════════
            ClaimGroundingAnalyzer(self.config),
            ClaimAwareSufficiencyAnalyzer(self.config),
            CitationFaithfulnessProbe(self.config),
            # ════════════════════════════════════════════════════════════════
            # LAYER 4 — SECURITY
            # ════════════════════════════════════════════════════════════════
            PromptInjectionAnalyzer(self.config),
            PoisoningHeuristicAnalyzer(self.config),
            RetrievalAnomalyAnalyzer(self.config),
            PrivacyAnalyzer(self.config),
            # ════════════════════════════════════════════════════════════════
            # LAYER 5 — ATTRIBUTION + CONFIDENCE
            # ════════════════════════════════════════════════════════════════
            Layer6TaxonomyClassifier(self.config),
            SemanticEntropyAnalyzer(self.config),  # Always run (LLM or deterministic)
        ]

        # Optional analyzers (enabled via config flags)
        if self.config.get("enable_ncv", False):
            # Insert after grounding + claim-aware sufficiency + citation faithfulness, before security
            analyzers.insert(8, NCVPipelineVerifier(self.config))

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

    def _primary_failure(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
    ) -> FailureType:
        ranked_failures = self._ranked_failure_types(results, result_weights)
        return ranked_failures[0][0] if ranked_failures else FailureType.CLEAN

    def _primary_result(
        self,
        results: list[AnalyzerResult],
        primary_failure: FailureType,
        result_weights: dict[str, float],
    ) -> AnalyzerResult | None:
        if primary_failure == FailureType.CLEAN:
            return None
        return self._best_failure_results(results, result_weights).get(primary_failure)

    def _secondary_failures(
        self,
        results: list[AnalyzerResult],
        primary_failure: FailureType,
        result_weights: dict[str, float],
    ) -> list[FailureType]:
        return [
            failure_type
            for failure_type, _, _ in self._ranked_candidate_types(
                results,
                result_weights,
                statuses={"fail", "warn"},
            )
            if failure_type != primary_failure
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
        # Use SemanticEntropyAnalyzer score (works in both LLM and deterministic modes)
        for result in results:
            if result.analyzer_name == "SemanticEntropyAnalyzer":
                return result.score
        return None

    def _weighted_prior_results(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
    ) -> list[AnalyzerResult]:
        indexed_results = list(enumerate(results))
        sorted_results = sorted(
            indexed_results,
            key=lambda item: (
                -self._result_weight(item[1], result_weights),
                self.STATUS_ORDER.get(item[1].status, 99),
                item[0],
            ),
        )
        return [result for _, result in sorted_results]

    def _result_weight(
        self,
        result: AnalyzerResult,
        result_weights: dict[str, float],
    ) -> float:
        return result_weights.get(result.analyzer_name, 1.0)

    def _best_failure_results(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
    ) -> dict[FailureType, AnalyzerResult]:
        best_by_type: dict[FailureType, tuple[float, int, AnalyzerResult]] = {}
        for index, result in enumerate(results):
            if (
                result.status != "fail"
                or result.failure_type is None
                or result.analyzer_name in self.META_ANALYZER_NAMES
            ):
                continue
            weight = self._result_weight(result, result_weights)
            current = best_by_type.get(result.failure_type)
            if current is None or weight > current[0] or (weight == current[0] and index < current[1]):
                best_by_type[result.failure_type] = (weight, index, result)
        return {
            failure_type: payload[2]
            for failure_type, payload in best_by_type.items()
        }

    def _ranked_candidate_types(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
        *,
        statuses: set[str],
    ) -> list[tuple[FailureType, float, int]]:
        best_by_type: dict[FailureType, tuple[float, int, AnalyzerResult]] = {}
        for index, result in enumerate(results):
            if (
                result.status not in statuses
                or result.failure_type is None
                or result.analyzer_name in self.META_ANALYZER_NAMES
            ):
                continue
            weight = self._result_weight(result, result_weights)
            current = best_by_type.get(result.failure_type)
            if current is None or weight > current[0] or (weight == current[0] and index < current[1]):
                best_by_type[result.failure_type] = (weight, index, result)
        ranked = [
            (failure_type, payload[0], payload[1])
            for failure_type, payload in best_by_type.items()
        ]
        return self._sorted_ranked_failures(ranked)

    def _ranked_failure_types(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
    ) -> list[tuple[FailureType, float, int]]:
        return self._ranked_candidate_types(results, result_weights, statuses={"fail"})

    def _sorted_ranked_failures(
        self,
        ranked: list[tuple[FailureType, float, int]],
    ) -> list[tuple[FailureType, float, int]]:
        priority_index = {
            failure_type: index
            for index, failure_type in enumerate(FAILURE_PRIORITY)
        }
        ranked.sort(
            key=lambda item: (
                -item[1],
                priority_index.get(item[0], len(FAILURE_PRIORITY)),
                item[2],
            )
        )
        return ranked


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
