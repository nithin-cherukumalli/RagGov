"""Layer6 AI / TD Bank failure taxonomy classifier.

Classifies RAG failures into production failure modes across all pipeline stages:
PARSING, CHUNKING, RETRIEVAL, RERANKING, GENERATION, and SECURITY.

Based on: "Classifying and Addressing the Diversity of Errors in RAG Systems" (2025)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from raggov.models.grounding import GroundingEvidenceBundle

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


# Retrieval quality bands for cosine similarity over retrieved chunks.
# These are calibrated ranges, not paper-proven absolutes:
# - Scores above 0.75 are treated as likely relevant.
# - Scores below 0.60 are treated as likely irrelevant.
# - Scores in 0.60-0.75 remain ambiguous and should not be forced into a single cause.
SCORE_BAND_HIGH = 0.75
SCORE_BAND_LOW = 0.60


@dataclass
class StageFailure:
    """Represents a failure at a specific RAG pipeline stage."""

    stage: str  # PARSING | CHUNKING | RETRIEVAL | RERANKING | GENERATION | SECURITY
    failure_mode: str  # specific failure within that stage
    evidence: list[str]
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW


@dataclass
class Layer6FailureReport:
    """Complete Layer6 taxonomy classification report."""

    stage_failures: list[StageFailure]
    primary_stage: str
    failure_chain: list[str]  # ordered list showing how failure propagated
    engineer_action: str  # one concrete thing to fix


@dataclass
class WeightedStageFailure:
    """Internal stage failure paired with source authority metadata."""

    failure: StageFailure
    weight: float
    order: int
    remediation: str | None = None


# Engineer action recommendations for each failure mode
ENGINEER_ACTIONS = {
    # PARSING failures
    "lost_structure": "Use a structure-aware parser (unstructured.io, docling) before chunking.",
    "metadata_loss": "Verify parser preserves document identifiers and section metadata.",
    # RETRIEVAL failures
    "missing_relevant_docs": "Increase retrieval top-k and review embedding model quality.",
    "off_topic_retrieval": "Audit embedding model on your domain. Add query expansion or HyDE.",
    "stale_docs": "Re-index corpus. Add document freshness metadata and retrieval-time filtering.",
    "top_k_too_small": "Increase top-k to at least 5-8. Add MMR for diversity.",
    # CHUNKING failures
    "boundary_errors": "Review chunking strategy. Use semantic chunking or smaller fixed chunks with overlap.",
    "oversized_chunks": "Reduce chunk size. Target 256-512 tokens with 10-15% overlap.",
    # GENERATION failures
    "context_ignored": "Add explicit grounding instructions to system prompt. Consider RAG-specific fine-tuning.",
    "over_extraction": "Improve chunk quality and reranking. Add citation verification post-generation.",
    "hallucination": "Improve retrieval recall. Add RefChecker-style claim verification post-generation.",
    "mixed_quality": "Ambiguous: could be retrieval miss or context ignored. Check reranker quality and grounding prompt strength.",
    # SECURITY failures
    "prompt_injection": "Add input sanitization layer. Scan corpus for instruction-like content at ingest time.",
    "corpus_poisoning": "Audit corpus for anomalous documents. Add retrieval score monitoring.",
}

# Severity mapping based on failure type
SEVERITY_MAP = {
    FailureType.TABLE_STRUCTURE_LOSS: "HIGH",
    FailureType.HIERARCHY_FLATTENING: "MEDIUM",
    FailureType.METADATA_LOSS: "LOW",
    FailureType.PROMPT_INJECTION: "CRITICAL",
    FailureType.PRIVACY_VIOLATION: "CRITICAL",
    FailureType.RETRIEVAL_ANOMALY: "HIGH",
    FailureType.SUSPICIOUS_CHUNK: "HIGH",
    FailureType.CONTRADICTED_CLAIM: "HIGH",
    FailureType.UNSUPPORTED_CLAIM: "HIGH",
    FailureType.STALE_RETRIEVAL: "MEDIUM",
    FailureType.SCOPE_VIOLATION: "MEDIUM",
    FailureType.INSUFFICIENT_CONTEXT: "MEDIUM",
    FailureType.INCONSISTENT_CHUNKS: "MEDIUM",
    FailureType.CITATION_MISMATCH: "LOW",
}

# Stage ordering for failure chain prioritization
STAGE_ORDER = {
    "PARSING": 0,
    "CHUNKING": 1,
    "EMBEDDING": 2,
    "RETRIEVAL": 3,
    "RERANKING": 4,
    "GROUNDING": 5,
    "SUFFICIENCY": 6,
    "GENERATION": 7,
    "SECURITY": 8,
    "CONFIDENCE": 9,
}

NCV_STAGE_FAILURES = {
    "QUERY_UNDERSTANDING": ("RETRIEVAL", "off_topic_retrieval"),
    "RETRIEVAL_QUALITY": ("RETRIEVAL", "missing_relevant_docs"),
    "CONTEXT_ASSEMBLY": ("CHUNKING", "boundary_errors"),
    "CLAIM_GROUNDING": ("GENERATION", "hallucination"),
    "ANSWER_COMPLETENESS": ("GENERATION", "hallucination"),
}


class Layer6TaxonomyClassifier(BaseAnalyzer):
    """Maps prior analyzer results to Layer6 failure taxonomy."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze a RAG run and classify failures into Layer6 taxonomy."""
        weighted_mode = bool(self.config.get("weighted_prior_results"))
        prior_results = self.config.get("weighted_prior_results") or self.config.get("prior_results", [])
        analyzer_weights = self.config.get("analyzer_weights", {})
        ncv_report = self.config.get("ncv_report")

        failed_results = [r for r in prior_results if r.status in ("fail", "warn")]
        if not failed_results:
            return self.skip("no prior failures to classify")

        chunk_scores = [
            chunk.score for chunk in run.retrieved_chunks if chunk.score is not None
        ]

        stage_failures: list[WeightedStageFailure] = []
        stage_failures.extend(self._detect_parsing_failures(prior_results, analyzer_weights))
        stage_failures.extend(self._detect_retrieval_failures(prior_results, chunk_scores, analyzer_weights))
        stage_failures.extend(self._detect_chunking_failures(prior_results, chunk_scores, analyzer_weights))
        
        # Extract grounding bundle if available
        bundle = self._get_grounding_bundle(prior_results)
        
        stage_failures.extend(self._detect_generation_failures(prior_results, chunk_scores, analyzer_weights, bundle))
        stage_failures.extend(self._detect_security_failures(prior_results, analyzer_weights))
        stage_failures = self._dedupe_failures(stage_failures)
        stage_failures = self._apply_ncv_tiebreaker(stage_failures, ncv_report, analyzer_weights)

        if not stage_failures:
            return self.skip("no Layer6 failure modes detected")

        failure_chain = self._build_failure_chain(stage_failures)
        primary_stage = self._identify_primary_stage(stage_failures)
        primary_failure = self._identify_primary_failure(stage_failures, primary_stage)
        engineer_action = self._engineer_action(primary_failure, prefer_source_remediation=weighted_mode)

        report = Layer6FailureReport(
            stage_failures=[failure.failure for failure in stage_failures],
            primary_stage=primary_stage,
            failure_chain=failure_chain,
            engineer_action=engineer_action,
        )

        report_dict = {
            "stage_failures": [asdict(f) for f in report.stage_failures],
            "primary_stage": report.primary_stage,
            "failure_chain": report.failure_chain,
            "engineer_action": report.engineer_action,
        }

        stage_map = {
            "PARSING": FailureStage.PARSING,
            "CHUNKING": FailureStage.CHUNKING,
            "EMBEDDING": FailureStage.EMBEDDING,
            "RETRIEVAL": FailureStage.RETRIEVAL,
            "RERANKING": FailureStage.RERANKING,
            "GENERATION": FailureStage.GENERATION,
            "SECURITY": FailureStage.SECURITY,
        }

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=None,
            stage=stage_map.get(primary_stage, FailureStage.UNKNOWN),
            evidence=[json.dumps(report_dict)],
            remediation=engineer_action,
        )

    def _detect_parsing_failures(
        self,
        prior_results: list[AnalyzerResult],
        analyzer_weights: dict[str, float],
    ) -> list[WeightedStageFailure]:
        """Detect PARSING stage failures."""
        failures: list[WeightedStageFailure] = []

        for order, result in enumerate(prior_results):
            if result.status not in ("fail", "warn") or result.failure_type is None:
                continue

            if result.failure_type in (
                FailureType.TABLE_STRUCTURE_LOSS,
                FailureType.HIERARCHY_FLATTENING,
            ):
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="PARSING",
                        failure_mode="lost_structure",
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )
            elif result.failure_type == FailureType.METADATA_LOSS:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="PARSING",
                        failure_mode="metadata_loss",
                        severity=SEVERITY_MAP.get(result.failure_type, "LOW"),
                    )
                )

        return self._dedupe_failures(failures)

    def _detect_retrieval_failures(
        self,
        prior_results: list[AnalyzerResult],
        chunk_scores: list[float],
        analyzer_weights: dict[str, float],
    ) -> list[WeightedStageFailure]:
        """Detect RETRIEVAL stage failures."""
        failures: list[WeightedStageFailure] = []
        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        for order, result in enumerate(prior_results):
            if result.status not in ("fail", "warn"):
                continue

            if result.failure_type == FailureType.SCOPE_VIOLATION:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="RETRIEVAL",
                        failure_mode="off_topic_retrieval",
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )
            elif result.failure_type == FailureType.STALE_RETRIEVAL:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="RETRIEVAL",
                        failure_mode="stale_docs",
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )
            elif result.failure_type == FailureType.INSUFFICIENT_CONTEXT:
                if avg_score < 0.65:
                    failures.append(
                        self._candidate(
                            result,
                            analyzer_weights,
                            order,
                            stage="RETRIEVAL",
                            failure_mode="missing_relevant_docs",
                            severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                        )
                    )
                elif len(chunk_scores) < 3:
                    score_variance = statistics.stdev(chunk_scores) if len(chunk_scores) > 1 else 0
                    if score_variance > 0.2:
                        failures.append(
                            self._candidate(
                                result,
                                analyzer_weights,
                                order,
                                stage="RETRIEVAL",
                                failure_mode="top_k_too_small",
                                severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                            )
                        )
                    else:
                        failures.append(
                            self._candidate(
                                result,
                                analyzer_weights,
                                order,
                                stage="RETRIEVAL",
                                failure_mode="top_k_too_small",
                                severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                            )
                        )
                else:
                    failures.append(
                        self._candidate(
                            result,
                            analyzer_weights,
                            order,
                            stage="RETRIEVAL",
                            failure_mode="missing_relevant_docs",
                            severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                        )
                    )

        return self._dedupe_failures(failures)

    def _detect_chunking_failures(
        self,
        prior_results: list[AnalyzerResult],
        chunk_scores: list[float],
        analyzer_weights: dict[str, float],
    ) -> list[WeightedStageFailure]:
        """Detect CHUNKING stage failures."""
        failures: list[WeightedStageFailure] = []

        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        for order, result in enumerate(prior_results):
            if result.status not in ("fail", "warn"):
                continue

            if result.failure_type == FailureType.INCONSISTENT_CHUNKS:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="CHUNKING",
                        failure_mode="boundary_errors",
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )

        return self._dedupe_failures(failures)

    def _detect_generation_failures(
        self,
        prior_results: list[AnalyzerResult],
        chunk_scores: list[float],
        analyzer_weights: dict[str, float],
        bundle: GroundingEvidenceBundle | None = None,
    ) -> list[WeightedStageFailure]:
        """Detect GENERATION stage failures.

        Uses GroundingEvidenceBundle for high-fidelity diagnosis if available.
        Otherwise falls back to avg_score heuristic bands.
        """
        failures: list[WeightedStageFailure] = []
        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        # High-fidelity diagnosis using bundle
        if bundle:
            for order, result in enumerate(prior_results):
                if result.status not in ("fail", "warn"):
                    continue
                
                if result.failure_type == FailureType.UNSUPPORTED_CLAIM:
                    # RAGChecker-style: check if any candidate had high score
                    rollup = bundle.diagnostic_rollup or {}
                    context_ignored_count = rollup.get("context_ignored_suspected_count", 0)
                    retrieval_miss_count = rollup.get("retrieval_miss_suspected_count", 0)
                    
                    if context_ignored_count > 0:
                        failures.append(self._candidate(result, analyzer_weights, order, stage="GENERATION", failure_mode="context_ignored", severity="HIGH"))
                    elif retrieval_miss_count > 0:
                        failures.append(self._candidate(result, analyzer_weights, order, stage="GENERATION", failure_mode="hallucination", severity="HIGH"))
                    else:
                        failures.append(self._candidate(result, analyzer_weights, order, stage="GENERATION", failure_mode="mixed_quality", severity="HIGH"))
                
                elif result.failure_type == FailureType.CONTRADICTED_CLAIM:
                    failures.append(self._candidate(result, analyzer_weights, order, stage="GENERATION", failure_mode="over_extraction", severity="HIGH"))

            # Check for value errors directly in records
            value_error_count = sum(1 for r in bundle.claim_evidence_records if r.uncertainty_signals.get("value_conflicts"))
            if value_error_count > 0:
                # Add a synthetic failure for value distortion if not already present
                failures.append(WeightedStageFailure(
                    failure=StageFailure(stage="GENERATION", failure_mode="value_distortion", evidence=[f"{value_error_count} claims had value conflicts."], severity="HIGH"),
                    weight=1.0, order=-1
                ))

            return self._dedupe_failures(failures)

        # Legacy heuristic path
        for order, result in enumerate(prior_results):
            if result.status not in ("fail", "warn"):
                continue

            if result.failure_type == FailureType.UNSUPPORTED_CLAIM and avg_score > SCORE_BAND_HIGH:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="GENERATION",
                        failure_mode="context_ignored",
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )
            elif result.failure_type == FailureType.UNSUPPORTED_CLAIM and avg_score <= SCORE_BAND_LOW:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="GENERATION",
                        failure_mode="hallucination",
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )
            elif (
                result.failure_type == FailureType.UNSUPPORTED_CLAIM
                and SCORE_BAND_LOW < avg_score <= SCORE_BAND_HIGH
            ):
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="GENERATION",
                        failure_mode="mixed_quality",
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )
            elif result.failure_type == FailureType.CONTRADICTED_CLAIM:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="GENERATION",
                        failure_mode="over_extraction",
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

        return self._dedupe_failures(failures)

    def _detect_security_failures(
        self,
        prior_results: list[AnalyzerResult],
        analyzer_weights: dict[str, float],
    ) -> list[WeightedStageFailure]:
        """Detect SECURITY stage failures."""
        failures: list[WeightedStageFailure] = []

        for order, result in enumerate(prior_results):
            if result.status not in ("fail", "warn"):
                continue

            if result.failure_type == FailureType.PROMPT_INJECTION:
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="SECURITY",
                        failure_mode="prompt_injection",
                        severity=SEVERITY_MAP.get(result.failure_type, "CRITICAL"),
                    )
                )
            elif result.failure_type in (
                FailureType.SUSPICIOUS_CHUNK,
                FailureType.RETRIEVAL_ANOMALY,
            ):
                failures.append(
                    self._candidate(
                        result,
                        analyzer_weights,
                        order,
                        stage="SECURITY",
                        failure_mode="corpus_poisoning",
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

        return self._dedupe_failures(failures)

    def _build_failure_chain(self, stage_failures: list[WeightedStageFailure]) -> list[str]:
        """Build ordered failure chain showing how failures propagated."""
        sorted_failures = sorted(
            stage_failures,
            key=lambda failure: (
                STAGE_ORDER.get(failure.failure.stage, 999),
                failure.order,
            ),
        )
        return [
            f"{failure.failure.stage} → {failure.failure.failure_mode}"
            for failure in sorted_failures
        ]

    def _identify_primary_stage(
        self,
        stage_failures: list[WeightedStageFailure],
    ) -> str:
        """Identify the primary stage using earliest causal precedence."""
        if not stage_failures:
            return "UNKNOWN"
        return sorted(
            stage_failures,
            key=lambda failure: (STAGE_ORDER.get(failure.failure.stage, 999), failure.order),
        )[0].failure.stage

    def _identify_primary_failure(
        self,
        stage_failures: list[WeightedStageFailure],
        primary_stage: str,
    ) -> WeightedStageFailure:
        """Identify the strongest failure within the chosen primary stage."""
        if not stage_failures:
            raise ValueError("stage_failures must not be empty")
        stage_specific_failures = [
            failure for failure in stage_failures if failure.failure.stage == primary_stage
        ]
        candidates = stage_specific_failures or stage_failures
        return sorted(
            candidates,
            key=lambda failure: (-failure.weight, failure.order),
        )[0]

    def _apply_ncv_tiebreaker(
        self,
        stage_failures: list[WeightedStageFailure],
        ncv_report: dict[str, Any] | None,
        analyzer_weights: dict[str, float],
    ) -> list[WeightedStageFailure]:
        """Add an earlier NCV bottleneck without overriding explicit parsing or security."""
        if not ncv_report:
            return stage_failures

        first_failing_node = ncv_report.get("first_failing_node")
        if first_failing_node not in NCV_STAGE_FAILURES:
            return stage_failures

        if any(failure.failure.stage == "PARSING" for failure in stage_failures):
            return stage_failures
        if any(failure.failure.stage == "SECURITY" for failure in stage_failures):
            return stage_failures

        ncv_stage, failure_mode = NCV_STAGE_FAILURES[first_failing_node]
        current_primary_stage = (
            self._identify_primary_stage(stage_failures)
            if stage_failures
            else "UNKNOWN"
        )
        if stage_failures and STAGE_ORDER.get(ncv_stage, 999) >= STAGE_ORDER.get(current_primary_stage, 999):
            return stage_failures
        ncv_failure = WeightedStageFailure(
            failure=StageFailure(
                stage=ncv_stage,
                failure_mode=failure_mode,
                evidence=[
                    ncv_report.get(
                        "bottleneck_description",
                        "NCV identified an earlier-stage pipeline bottleneck",
                    )
                ],
                severity="MEDIUM",
            ),
            weight=float(analyzer_weights.get("NCVPipelineVerifier", 1.0)),
            order=-1,
        )
        return self._dedupe_failures([ncv_failure, *stage_failures])

    def _engineer_action(
        self,
        primary_failure: WeightedStageFailure,
        *,
        prefer_source_remediation: bool,
    ) -> str:
        """Choose engineer action, preferring the strongest supporting remediation."""
        if primary_failure.failure.stage == "PARSING" and primary_failure.remediation:
            return primary_failure.remediation
        if prefer_source_remediation and primary_failure.remediation:
            return primary_failure.remediation
        return ENGINEER_ACTIONS.get(
            primary_failure.failure.failure_mode,
            "Review retrieval and generation pipeline for issues.",
        )

    def _candidate(
        self,
        result: AnalyzerResult,
        analyzer_weights: dict[str, float],
        order: int,
        *,
        stage: str,
        failure_mode: str,
        severity: str,
    ) -> WeightedStageFailure:
        return WeightedStageFailure(
            failure=StageFailure(
                stage=stage,
                failure_mode=failure_mode,
                evidence=result.evidence,
                severity=severity,
            ),
            weight=self._result_weight(result, analyzer_weights),
            order=order,
            remediation=result.remediation,
        )

    def _dedupe_failures(
        self,
        stage_failures: list[WeightedStageFailure],
    ) -> list[WeightedStageFailure]:
        best_failures: dict[tuple[str, str], WeightedStageFailure] = {}
        for failure in stage_failures:
            key = (failure.failure.stage, failure.failure.failure_mode)
            current = best_failures.get(key)
            if current is None or failure.weight > current.weight or (
                failure.weight == current.weight and failure.order < current.order
            ):
                best_failures[key] = failure
        return list(best_failures.values())

    def _result_weight(
        self,
        result: AnalyzerResult,
        analyzer_weights: dict[str, float],
    ) -> float:
        return float(analyzer_weights.get(result.analyzer_name, 1.0))

    def _get_grounding_bundle(self, prior_results: list[AnalyzerResult]) -> GroundingEvidenceBundle | None:
        """Extract the grounding bundle from prior results."""
        for result in prior_results:
            if getattr(result, "grounding_evidence_bundle", None):
                return result.grounding_evidence_bundle
        return None
