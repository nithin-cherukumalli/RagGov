"""Core orchestration engine for running RagGov analyses."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer
from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0
from raggov.analyzers.confidence.semantic_entropy import SemanticEntropyAnalyzer
from raggov.analyzers.grounding.citation_faithfulness import CitationFaithfulnessProbe
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.parsing.parser_validation import ParserValidationAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.evidence_profile import RetrievalEvidenceProfilerV0
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
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
from raggov.analyzers.version_validity import (
    TemporalSourceValidityAnalyzerV1,
    VersionValidityAnalyzerV1,
)
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
from raggov.evaluators.base import ExternalEvaluationResult
from raggov.evaluators.registry import create_standard_registry


logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """Run analyzers and merge their outputs into a diagnosis."""

    META_ANALYZER_NAMES = {
        "Layer6TaxonomyClassifier",
        "A2PAttributionAnalyzer",
        "CitationFaithfulnessAnalyzerV0",
        "TemporalSourceValidityAnalyzerV1",
        "VersionValidityAnalyzerV1",
        "RetrievalDiagnosisAnalyzerV0",
    }
    STATUS_ORDER = {
        "fail": 0,
        "warn": 1,
        "pass": 2,
        "skip": 3,
    }
    CRITICAL_ANALYZER_TYPES = (
        ClaimGroundingAnalyzer,
        CitationFaithfulnessAnalyzerV0,
        CitationFaithfulnessProbe,
        RetrievalDiagnosisAnalyzerV0,
        NCVPipelineVerifier,
        A2PAttributionAnalyzer,
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        analyzers: Sequence[BaseAnalyzer] | None = None,
    ) -> None:
        self.config = config or {}
        
        mode = self.config.get("mode", "external-enhanced")
        self.config["mode"] = mode
        
        if mode == "calibrated":
            raise NotImplementedError("Calibrated mode is not yet available natively.")
            
        enabled_providers = self.config.get("enabled_external_providers", [])
        if mode == "external-enhanced" and "enabled_external_providers" not in self.config:
            enabled_providers = [
                "structured_llm_claim", "structured_llm_citation",
                "cross_encoder_relevance", "ragas", "deepeval",
                "refchecker_claim", "refchecker_citation", "ragchecker",
                "a2p",
            ]
            self.config["enabled_external_providers"] = enabled_providers
            
        has_llm = bool(self.config.get("llm_client")) or bool(self.config.get("llm_fn"))
        
        if mode == "native":
            self.config["claim_verifier"] = "heuristic"
            self.config["citation_verifier"] = "native"
            self.config["retrieval_relevance_provider"] = "native"
            self.config["enable_a2p"] = False
            self.config["enabled_external_providers"] = []
        else:
            if "structured_llm_claim" in enabled_providers:
                self.config.setdefault("claim_verifier", "structured_llm")
            if "structured_llm_citation" in enabled_providers:
                self.config.setdefault("citation_verifier", "structured_llm")
            if "refchecker_claim" in enabled_providers:
                self.config.setdefault("claim_verifier", self.config.get("claim_verifier", "refchecker"))
            if "refchecker_citation" in enabled_providers:
                self.config.setdefault("citation_verifier", self.config.get("citation_verifier", "refchecker"))
            if "cross_encoder_relevance" in enabled_providers:
                self.config.setdefault("retrieval_relevance_provider", "cross_encoder")
            if "a2p" in enabled_providers:
                self.config.setdefault("enable_a2p", True)
                
        self.external_registry = create_standard_registry(self.config)
        self.analyzers = list(analyzers) if analyzers is not None else self._default_analyzers()

    def diagnose(self, run: RAGRun) -> Diagnosis:
        """Run the research-backed 5-layer analyzer suite and return merged diagnosis."""
        results: list[AnalyzerResult] = []
        result_weights: dict[str, float] = {}

        mode = self.config.get("mode", "external-enhanced")
        enabled_providers = self.config.get("enabled_external_providers", [])
        has_llm = bool(self.config.get("llm_client")) or bool(self.config.get("llm_fn"))
        
        missing_external_providers = []
        external_signals_used = []
        external_adapter_errors = []
        fallback_heuristics_used = []
        degraded_external_mode = False

        if mode == "external-enhanced":
            eval_providers_to_run = [p for p in enabled_providers if p not in ("structured_llm_claim", "structured_llm_citation")]
            if self.config.get("retrieval_relevance_provider") == "native" and "cross_encoder_relevance" in eval_providers_to_run:
                eval_providers_to_run.remove("cross_encoder_relevance")
            eval_results = self.external_registry.evaluate_enabled(
                run,
                eval_providers_to_run,
                strict_mode=self.config.get("strict_external_evaluators", False),
            )
            
            if self.config.get("claim_verifier") == "structured_llm" and not has_llm:
                if "structured_llm_claim" not in missing_external_providers:
                    missing_external_providers.append("structured_llm_claim")
                degraded_external_mode = True

            if self.config.get("citation_verifier") == "structured_llm" and not has_llm:
                if "structured_llm_citation" not in missing_external_providers:
                    missing_external_providers.append("structured_llm_citation")
                degraded_external_mode = True

            run.metadata["external_evaluation_results"] = [r.model_dump() for r in eval_results]
            
            for r in eval_results:
                name = r.adapter_name or str(r.provider.value)
                if r.succeeded:
                    external_signals_used.append(name)
                elif r.missing_dependency:
                    missing_external_providers.append(name)
                    degraded_external_mode = True
                else:
                    # In non-strict mode, some errors might be considered missing providers 
                    # if they are due to environmental reasons.
                    if "not installed" in (r.error or "").lower():
                        missing_external_providers.append(name)
                        degraded_external_mode = True
                    else:
                        external_adapter_errors.append(f"{name}: {r.error}")

        # Maintain legacy a2p check
        if "a2p" in enabled_providers and mode != "native":
            if not has_llm:
                if "a2p" not in missing_external_providers:
                    missing_external_providers.append("a2p")
                if "legacy_failure_level_heuristic" not in fallback_heuristics_used:
                    fallback_heuristics_used.append("legacy_failure_level_heuristic")
                degraded_external_mode = True
            else:
                if "a2p" not in external_signals_used:
                    external_signals_used.append("a2p")

        # Map external signals/missing providers to fallbacks for backward compatibility
        if "structured_llm_claim" in missing_external_providers:
            if "heuristic_claim_verifier" not in fallback_heuristics_used:
                fallback_heuristics_used.append("heuristic_claim_verifier")

        if "cross_encoder_relevance" in missing_external_providers:
            if "lexical_overlap_relevance" not in fallback_heuristics_used:
                fallback_heuristics_used.append("lexical_overlap_relevance")

        if "ragas" in missing_external_providers:
            if "native_retrieval_signals_only" not in fallback_heuristics_used:
                fallback_heuristics_used.append("native_retrieval_signals_only")

        if "deepeval" in missing_external_providers:
            if "native_retrieval_signals_only" not in fallback_heuristics_used:
                fallback_heuristics_used.append("native_retrieval_signals_only")

        if "ragchecker" in missing_external_providers:
            if "native_retrieval_signals_only" not in fallback_heuristics_used:
                fallback_heuristics_used.append("native_retrieval_signals_only")

        if "refchecker_claim" in missing_external_providers:
            if "heuristic_claim_verifier" not in fallback_heuristics_used:
                fallback_heuristics_used.append("heuristic_claim_verifier")

        if "refchecker_citation" in missing_external_providers:
            if "native_citation_verifier" not in fallback_heuristics_used:
                fallback_heuristics_used.append("native_citation_verifier")

        if mode == "external-enhanced" and self.config.get("strict_external_evaluators"):
            if missing_external_providers or external_adapter_errors:
                raise RuntimeError(
                    f"Strict mode enabled but external providers missing or failed: "
                    f"missing={missing_external_providers}, errors={external_adapter_errors}"
                )

        run.metadata["external_signals_used"] = external_signals_used
        run.metadata["missing_external_providers"] = missing_external_providers
        run.metadata["external_adapter_errors"] = external_adapter_errors
        run.metadata["degraded_external_mode"] = degraded_external_mode
        run.metadata["fallback_heuristics_used"] = fallback_heuristics_used

        for analyzer in self.analyzers:
            if isinstance(
                analyzer,
                (
                    ScopeViolationAnalyzer,
                    StaleRetrievalAnalyzer,
                    CitationMismatchAnalyzer,
                    InconsistentChunksAnalyzer,
                    CitationFaithfulnessAnalyzerV0,
                    TemporalSourceValidityAnalyzerV1,
                    VersionValidityAnalyzerV1,
                ),
            ):
                self._ensure_retrieval_evidence_profile(run)

            # Pass prior results to aggregation analyzers (Layer 5)
            # These analyzers need context from all prior detection layers
            if isinstance(
                analyzer,
                (
                    A2PAttributionAnalyzer,
                    Layer6TaxonomyClassifier,
                    SemanticEntropyAnalyzer,
                    ClaimAwareSufficiencyAnalyzer,
                    RetrievalDiagnosisAnalyzerV0,
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
            self._attach_result_outputs(run, result)

        primary_failure, missing_critical = self._primary_failure(results, result_weights, run)
        primary_result = self._primary_result(results, primary_failure, result_weights)
        secondary_failures = self._secondary_failures(results, primary_failure, result_weights)

        # Check if A2P ran and produced an attribution
        a2p_result = next(
            (r for r in results if r.analyzer_name == "A2PAttributionAnalyzer" and r.status == "fail"),
            None,
        )

        # Only allow A2P to override stage when it carries non-legacy, evidence-backed claim attribution.
        if (
            primary_result is not None
            and primary_result.stage == FailureStage.PARSING
        ):
            root_cause_stage = FailureStage.PARSING
        elif self._should_use_a2p_stage_override(a2p_result) and a2p_result is not None:
            root_cause_stage = a2p_result.attribution_stage or FailureStage.UNKNOWN
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
        first_uncertain_node = None
        ncv_bottleneck_description = None
        ncv_downstream_failure_chain = []
        ncv_missing_reports = []
        ncv_fallback_heuristics_used = []

        if ncv_result is not None and ncv_result.evidence:
            try:
                ncv_report = json.loads(ncv_result.evidence[0])
                pipeline_health_score = ncv_report.get("pipeline_health_score")
                first_failing_node = ncv_report.get("first_failing_node")
                first_uncertain_node = ncv_report.get("first_uncertain_node")
                ncv_bottleneck_description = ncv_report.get("bottleneck_description")
                ncv_downstream_failure_chain = ncv_report.get("downstream_failure_chain", [])
                ncv_missing_reports = ncv_report.get("missing_reports", [])
                ncv_fallback_heuristics_used = ncv_report.get("fallback_heuristics_used", [])
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
        citation_faithfulness_v0_result = next(
            (r for r in results if r.analyzer_name == "CitationFaithfulnessAnalyzerV0"),
            None,
        )
        citation_faithfulness_report = (
            citation_faithfulness_v0_result.citation_faithfulness_report
            if citation_faithfulness_v0_result is not None
            else None
        )
        citation_faithfulness = "unchecked"
        if citation_faithfulness_result is not None:
            if citation_faithfulness_result.status == "pass":
                citation_faithfulness = "genuine"
            elif citation_faithfulness_result.status == "fail":
                citation_faithfulness = "post_rationalized"
            elif citation_faithfulness_result.status == "warn":
                citation_faithfulness = "partial"

        version_validity_result = next(
            (
                r
                for r in results
                if r.analyzer_name
                in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1"}
            ),
            None,
        )
        version_validity_report = (
            version_validity_result.version_validity_report
            if version_validity_result is not None
            else None
        )
        retrieval_diagnosis_result = next(
            (r for r in results if r.analyzer_name == "RetrievalDiagnosisAnalyzerV0"),
            None,
        )
        retrieval_diagnosis_report = (
            retrieval_diagnosis_result.retrieval_diagnosis_report
            if retrieval_diagnosis_result is not None
            else None
        )

        grounding_result = next(
            (r for r in results if r.analyzer_name == "ClaimGroundingAnalyzer"),
            None,
        )
        diagnosis_claim_results = grounding_result.claim_results or [] if grounding_result else []

        diagnosis = Diagnosis(
            run_id=run.run_id,
            diagnosis_mode=mode,
            external_signals_used=external_signals_used,
            missing_external_providers=missing_external_providers,
            fallback_heuristics_used=fallback_heuristics_used,
            degraded=degraded_external_mode,
            degraded_external_mode=degraded_external_mode,
            external_adapter_errors=external_adapter_errors,
            primary_failure=primary_failure,
            secondary_failures=secondary_failures,
            root_cause_stage=root_cause_stage,
            should_have_answered=should_have_answered(primary_failure),
            security_risk=_max_risk(results),
            confidence=confidence,
            claim_results=diagnosis_claim_results,
            evidence=self._evidence(results) + missing_critical,
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
            first_uncertain_node=first_uncertain_node,
            ncv_bottleneck_description=ncv_bottleneck_description,
            ncv_downstream_failure_chain=ncv_downstream_failure_chain,
            ncv_missing_reports=ncv_missing_reports,
            ncv_fallback_heuristics_used=ncv_fallback_heuristics_used,
            citation_faithfulness=citation_faithfulness,
            citation_faithfulness_report=citation_faithfulness_report,
            version_validity_report=version_validity_report,
            retrieval_diagnosis_report=retrieval_diagnosis_report,
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

        for attribution_v2 in a2p_result.claim_attributions_v2 or []:
            if attribution_v2.fallback_used:
                continue
            if attribution_v2.attribution_method == "legacy_failure_level_heuristic":
                continue
            if attribution_v2.evidence_summary:
                return True
            if any(candidate.evidence_for for candidate in attribution_v2.candidate_causes):
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
            # LAYER 3 — GROUNDING
            # ════════════════════════════════════════════════════════════════
            ClaimGroundingAnalyzer(self.config),
            ClaimAwareSufficiencyAnalyzer(self.config),
            # ════════════════════════════════════════════════════════════════
            # LAYER 2 — RETRIEVAL HEALTH
            # ════════════════════════════════════════════════════════════════
            ScopeViolationAnalyzer(self.config),
            StaleRetrievalAnalyzer(self.config),
            CitationMismatchAnalyzer(self.config),
            InconsistentChunksAnalyzer(self.config),
            CitationFaithfulnessAnalyzerV0(self.config),
            TemporalSourceValidityAnalyzerV1(self.config),
            RetrievalDiagnosisAnalyzerV0(self.config),
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
        enable_ncv = self.config.get("enable_ncv")
        if enable_ncv is None:
            # Enable by default in external-enhanced mode
            enable_ncv = (self.config.get("mode") == "external-enhanced")

        if enable_ncv:
            # Insert after grounding + claim-aware sufficiency + citation faithfulness, before security
            analyzers.insert(12, NCVPipelineVerifier(self.config))

        if self.config.get("enable_a2p", False):
            analyzers.append(A2PAttributionAnalyzer(self.config))

        return analyzers

    def _run_analyzer(self, analyzer: BaseAnalyzer, run: RAGRun) -> AnalyzerResult:
        try:
            return analyzer.analyze(run)
        except Exception as exc:
            logger.exception("Analyzer %s failed", analyzer.name())
            if isinstance(analyzer, self.CRITICAL_ANALYZER_TYPES):
                error_payload = {
                    "event": "critical_analyzer_error",
                    "analyzer": analyzer.name(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                return AnalyzerResult(
                    analyzer_name=analyzer.name(),
                    status="fail",
                    failure_type=FailureType.INCOMPLETE_DIAGNOSIS,
                    stage=FailureStage.UNKNOWN,
                    evidence=[json.dumps(error_payload)],
                    remediation=DEFAULT_REMEDIATIONS[FailureType.INCOMPLETE_DIAGNOSIS],
                )
            return AnalyzerResult(
                analyzer_name=analyzer.name(),
                status="skip",
                evidence=[str(exc)],
            )

    def _ensure_retrieval_evidence_profile(self, run: RAGRun) -> None:
        """Populate retrieval evidence once for analyzers that consume it."""
        if run.retrieval_evidence_profile is not None:
            return
            
        profiler = RetrievalEvidenceProfilerV0(self.config)
        profile = profiler.build(run)
        
        # Apply external signals if available in run.metadata
        eval_results_raw = run.metadata.get("external_evaluation_results", [])
        signals = []
        for item in eval_results_raw:
            if isinstance(item, dict):
                try:
                    res = ExternalEvaluationResult.model_validate(item)
                    signals.extend(res.signals)
                except Exception:
                    continue
            elif isinstance(item, ExternalEvaluationResult):
                signals.extend(item.signals)
                
        run.retrieval_evidence_profile = profiler.apply_external_relevance_signals(profile, signals)
        run.metadata["retrieval_evidence_profile_used"] = True

    def _attach_result_outputs(self, run: RAGRun, result: AnalyzerResult) -> None:
        """Expose structured analyzer outputs to downstream analyzers."""
        if result.grounding_evidence_bundle is not None:
            run.metadata["grounding_evidence_bundle"] = result.grounding_evidence_bundle
            run.metadata["claim_evidence_records"] = (
                result.grounding_evidence_bundle.claim_evidence_records
            )
        if result.citation_faithfulness_report is not None:
            run.citation_faithfulness_report = result.citation_faithfulness_report
        if result.version_validity_report is not None:
            run.version_validity_report = result.version_validity_report
        if result.retrieval_diagnosis_report is not None:
            run.retrieval_diagnosis_report = result.retrieval_diagnosis_report

    def _primary_failure(
        self,
        results: list[AnalyzerResult],
        result_weights: dict[str, float],
        run: RAGRun,
    ) -> tuple[FailureType, list[str]]:
        ranked_failures = self._ranked_failure_types(results, result_weights)
        primary = ranked_failures[0][0] if ranked_failures else FailureType.CLEAN
        
        # Reliability Gate: Prevent false CLEAN if critical evidence is missing
        if primary == FailureType.CLEAN:
            missing_critical = self._get_missing_critical_evidence(results, run)
            if missing_critical:
                return FailureType.INCOMPLETE_DIAGNOSIS, missing_critical
                
        return primary, []

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

    def _get_missing_critical_evidence(
        self, results: list[AnalyzerResult], run: RAGRun
    ) -> list[str]:
        """Check if mandatory analyzers for the current mode completed successfully."""
        mode = self.config.get("mode", "external-enhanced")
        completed_names = {r.analyzer_name for r in results if r.status in {"pass", "warn"}}
        
        # Analyzers that failed with INCOMPLETE_DIAGNOSIS (crashed) are not "completed"
        failed_critical = {
            r.analyzer_name for r in results 
            if r.status == "fail" and r.failure_type == FailureType.INCOMPLETE_DIAGNOSIS
        }
        completed_names -= failed_critical

        missing = []
        
        # 1. ClaimGrounding is always critical
        if "ClaimGroundingAnalyzer" not in completed_names:
            if not (
                mode == "native"
                and self._has_acceptable_native_claim_grounding_skip(results)
            ):
                missing.append("ClaimGroundingAnalyzer")
            
        # 2. RetrievalDiagnosis is critical in enhanced, optional in native (but checked if present)
        if mode == "external-enhanced":
            if "RetrievalDiagnosisAnalyzerV0" not in completed_names:
                missing.append("RetrievalDiagnosisAnalyzerV0")
            if "NCVPipelineVerifier" not in completed_names:
                missing.append("NCVPipelineVerifier")
        else:
            # Native mode: only critical if specifically enabled in analyzers list
            # Since we use default_analyzers, it's always there unless overridden.
            # User requirement: RetrievalDiagnosisAnalyzer result if retrieval diagnosis enabled
            # We'll assume it's enabled if it's in the suite.
            if any(isinstance(a, RetrievalDiagnosisAnalyzerV0) for a in self.analyzers):
                if "RetrievalDiagnosisAnalyzerV0" not in completed_names:
                    missing.append("RetrievalDiagnosisAnalyzerV0")

        # 3. Citation Faithfulness
        if mode == "external-enhanced":
            # Critical unless answer has no citations
            if run.cited_doc_ids and "CitationFaithfulnessAnalyzerV0" not in completed_names:
                missing.append("CitationFaithfulnessAnalyzerV0")
        elif self._native_citation_faithfulness_required(run):
            if "CitationFaithfulnessAnalyzerV0" not in completed_names:
                missing.append("CitationFaithfulnessAnalyzerV0")

        # 4. Parser Validation (if enabled)
        if any(isinstance(a, ParserValidationAnalyzer) for a in self.analyzers):
            if "ParserValidationAnalyzer" not in completed_names:
                missing.append("ParserValidationAnalyzer")

        # 5. Version Validity (if metadata or cited docs exist)
        if mode == "external-enhanced":
            has_version_context = bool(run.metadata.get("corpus_metadata")) or bool(run.cited_doc_ids)
            if has_version_context:
                if "VersionValidityAnalyzerV1" not in completed_names and "TemporalSourceValidityAnalyzerV1" not in completed_names:
                    missing.append("VersionValidityAnalyzerV1")

        # 6. External Providers (if configured in enhanced mode)
        if mode == "external-enhanced":
            expected_providers = self.config.get("enabled_external_providers", [])
            missing_providers = run.metadata.get("missing_external_providers", [])
            for provider in expected_providers:
                if provider in missing_providers:
                    # Check if this provider is considered critical (e.g. structured_llm_claim)
                    if provider in {"structured_llm_claim", "a2p"}:
                        missing.append(f"External Provider: {provider}")

        return missing

    def _has_acceptable_native_claim_grounding_skip(
        self,
        results: list[AnalyzerResult],
    ) -> bool:
        """Allow native CLEAN when claim extraction has no claim to verify.

        This is deliberately narrow: only the explicit "no claims extracted"
        skip is non-blocking in native mode. Crashes, unavailable analyzers,
        no retrieved chunks, and other skips still block CLEAN through the
        critical-evidence gate.
        """
        grounding = next(
            (result for result in results if result.analyzer_name == "ClaimGroundingAnalyzer"),
            None,
        )
        if grounding is None or grounding.status != "skip":
            return False
        return any("no claims extracted" in evidence for evidence in grounding.evidence)

    def _native_citation_faithfulness_required(self, run: RAGRun) -> bool:
        if bool(self.config.get("enable_citation_faithfulness")) or bool(
            self.config.get("citation_faithfulness_required")
        ):
            return True

        raw_records = run.metadata.get("claim_evidence_records")
        if raw_records:
            return bool(run.cited_doc_ids)

        bundle = run.metadata.get("grounding_evidence_bundle")
        if bundle is None:
            return False
        if hasattr(bundle, "claim_evidence_records"):
            return bool(run.cited_doc_ids and bundle.claim_evidence_records)
        if isinstance(bundle, dict):
            return bool(run.cited_doc_ids and bundle.get("claim_evidence_records"))
        return False


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
