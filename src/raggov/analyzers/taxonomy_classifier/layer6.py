"""Layer6 AI / TD Bank failure taxonomy classifier.

Classifies RAG failures into production failure modes across all pipeline stages:
PARSING, CHUNKING, RETRIEVAL, RERANKING, GENERATION, and SECURITY.

Based on: "Classifying and Addressing the Diversity of Errors in RAG Systems" (2025)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


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


# Engineer action recommendations for each failure mode
ENGINEER_ACTIONS = {
    # RETRIEVAL failures
    "missing_relevant_docs": "Increase retrieval top-k and review embedding model quality.",
    "off_topic_retrieval": "Audit embedding model on your domain. Add query expansion or HyDE.",
    "stale_docs": "Re-index corpus. Add document freshness metadata and retrieval-time filtering.",
    "top_k_too_small": "Increase top-k to at least 5-8. Add MMR for diversity.",
    # CHUNKING failures
    "boundary_errors": "Review chunking strategy. Use semantic chunking or smaller fixed chunks with overlap.",
    "lost_structure": "Use a structure-aware parser (unstructured.io, docling) before chunking.",
    "oversized_chunks": "Reduce chunk size. Target 256-512 tokens with 10-15% overlap.",
    # GENERATION failures
    "context_ignored": "Add explicit grounding instructions to system prompt. Consider RAG-specific fine-tuning.",
    "over_extraction": "Improve chunk quality and reranking. Add citation verification post-generation.",
    "hallucination": "Improve retrieval recall. Add RefChecker-style claim verification post-generation.",
    # SECURITY failures
    "prompt_injection": "Add input sanitization layer. Scan corpus for instruction-like content at ingest time.",
    "corpus_poisoning": "Audit corpus for anomalous documents. Add retrieval score monitoring.",
}

# Severity mapping based on failure type
SEVERITY_MAP = {
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


class Layer6TaxonomyClassifier(BaseAnalyzer):
    """Maps prior analyzer results to Layer6 failure taxonomy."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze a RAG run and classify failures into Layer6 taxonomy."""
        # Get prior results from config
        prior_results = self.config.get("prior_results", [])

        # Check if there are any failures to classify
        failed_results = [r for r in prior_results if r.status in ("fail", "warn")]
        if not failed_results:
            return self.skip("no prior failures to classify")

        # Extract chunk scores
        chunk_scores = [
            chunk.score for chunk in run.retrieved_chunks if chunk.score is not None
        ]

        # Detect failures for each stage
        stage_failures: list[StageFailure] = []
        stage_failures.extend(self._detect_retrieval_failures(prior_results, chunk_scores))
        stage_failures.extend(self._detect_chunking_failures(prior_results, chunk_scores))
        stage_failures.extend(self._detect_generation_failures(prior_results, chunk_scores))
        stage_failures.extend(self._detect_security_failures(prior_results))

        if not stage_failures:
            return self.skip("no Layer6 failure modes detected")

        # Build failure chain (ordered by stage)
        failure_chain = self._build_failure_chain(stage_failures)

        # Identify primary stage (earliest in chain)
        primary_stage = self._identify_primary_stage(stage_failures)

        # Get engineer action for primary failure
        primary_failure_mode = next(
            (f.failure_mode for f in stage_failures if f.stage == primary_stage), None
        )
        engineer_action = ENGINEER_ACTIONS.get(
            primary_failure_mode or "",
            "Review retrieval and generation pipeline for issues.",
        )

        # Build Layer6 report
        report = Layer6FailureReport(
            stage_failures=stage_failures,
            primary_stage=primary_stage,
            failure_chain=failure_chain,
            engineer_action=engineer_action,
        )

        # Serialize report to evidence as JSON
        report_dict = {
            "stage_failures": [asdict(f) for f in report.stage_failures],
            "primary_stage": report.primary_stage,
            "failure_chain": report.failure_chain,
            "engineer_action": report.engineer_action,
        }

        # Map primary stage back to FailureStage enum
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
            failure_type=None,  # Layer6 is meta-classification, not a specific failure
            stage=stage_map.get(primary_stage, FailureStage.UNKNOWN),
            evidence=[json.dumps(report_dict)],
            remediation=engineer_action,
        )

    def _detect_retrieval_failures(
        self, prior_results: list[AnalyzerResult], chunk_scores: list[float]
    ) -> list[StageFailure]:
        """Detect RETRIEVAL stage failures."""
        failures: list[StageFailure] = []

        # Calculate average chunk score
        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        for result in prior_results:
            if result.status not in ("fail", "warn"):
                continue

            # off_topic_retrieval: SCOPE_VIOLATION
            if result.failure_type == FailureType.SCOPE_VIOLATION:
                failures.append(
                    StageFailure(
                        stage="RETRIEVAL",
                        failure_mode="off_topic_retrieval",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )

            # stale_docs: STALE_RETRIEVAL
            elif result.failure_type == FailureType.STALE_RETRIEVAL:
                failures.append(
                    StageFailure(
                        stage="RETRIEVAL",
                        failure_mode="stale_docs",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )

            # missing_relevant_docs or top_k_too_small: INSUFFICIENT_CONTEXT
            elif result.failure_type == FailureType.INSUFFICIENT_CONTEXT:
                # Check if scores are low
                if avg_score < 0.65:
                    failures.append(
                        StageFailure(
                            stage="RETRIEVAL",
                            failure_mode="missing_relevant_docs",
                            evidence=result.evidence,
                            severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                        )
                    )
                # Check if top-k is too small (fewer than 3 chunks + varying scores)
                elif len(chunk_scores) < 3:
                    score_variance = statistics.stdev(chunk_scores) if len(chunk_scores) > 1 else 0
                    if score_variance > 0.2:  # Scores vary widely
                        failures.append(
                            StageFailure(
                                stage="RETRIEVAL",
                                failure_mode="top_k_too_small",
                                evidence=result.evidence,
                                severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                            )
                        )
                    else:
                        # Default to top_k_too_small if we have few chunks
                        failures.append(
                            StageFailure(
                                stage="RETRIEVAL",
                                failure_mode="top_k_too_small",
                                evidence=result.evidence,
                                severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                            )
                        )
                else:
                    # Default to missing_relevant_docs for other cases
                    failures.append(
                        StageFailure(
                            stage="RETRIEVAL",
                            failure_mode="missing_relevant_docs",
                            evidence=result.evidence,
                            severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                        )
                    )

        return failures

    def _detect_chunking_failures(
        self, prior_results: list[AnalyzerResult], chunk_scores: list[float]
    ) -> list[StageFailure]:
        """Detect CHUNKING stage failures."""
        failures: list[StageFailure] = []

        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        for result in prior_results:
            if result.status not in ("fail", "warn"):
                continue

            # boundary_errors: INCONSISTENT_CHUNKS
            if result.failure_type == FailureType.INCONSISTENT_CHUNKS:
                failures.append(
                    StageFailure(
                        stage="CHUNKING",
                        failure_mode="boundary_errors",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "MEDIUM"),
                    )
                )

            # lost_structure: UNSUPPORTED_CLAIM + moderate retrieval scores (0.65 < score <= 0.75)
            # Only detect chunking issue if scores are moderate (not very high)
            elif (
                result.failure_type == FailureType.UNSUPPORTED_CLAIM
                and 0.65 < avg_score <= 0.75
            ):
                failures.append(
                    StageFailure(
                        stage="CHUNKING",
                        failure_mode="lost_structure",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

            # oversized_chunks: low average chunk score despite non-empty retrieval
            # This is hard to detect from prior results alone, skip for now

        return failures

    def _detect_generation_failures(
        self, prior_results: list[AnalyzerResult], chunk_scores: list[float]
    ) -> list[StageFailure]:
        """Detect GENERATION stage failures."""
        failures: list[StageFailure] = []

        avg_score = statistics.mean(chunk_scores) if chunk_scores else 0.0

        for result in prior_results:
            if result.status not in ("fail", "warn"):
                continue

            # context_ignored: UNSUPPORTED_CLAIM + high retrieval scores (> 0.75)
            if result.failure_type == FailureType.UNSUPPORTED_CLAIM and avg_score > 0.75:
                failures.append(
                    StageFailure(
                        stage="GENERATION",
                        failure_mode="context_ignored",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

            # hallucination: UNSUPPORTED_CLAIM + low retrieval scores (<= 0.65)
            elif result.failure_type == FailureType.UNSUPPORTED_CLAIM and avg_score <= 0.65:
                failures.append(
                    StageFailure(
                        stage="GENERATION",
                        failure_mode="hallucination",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

            # over_extraction: CONTRADICTED_CLAIM
            elif result.failure_type == FailureType.CONTRADICTED_CLAIM:
                failures.append(
                    StageFailure(
                        stage="GENERATION",
                        failure_mode="over_extraction",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

        return failures

    def _detect_security_failures(
        self, prior_results: list[AnalyzerResult]
    ) -> list[StageFailure]:
        """Detect SECURITY stage failures."""
        failures: list[StageFailure] = []

        for result in prior_results:
            if result.status not in ("fail", "warn"):
                continue

            # prompt_injection: PROMPT_INJECTION
            if result.failure_type == FailureType.PROMPT_INJECTION:
                failures.append(
                    StageFailure(
                        stage="SECURITY",
                        failure_mode="prompt_injection",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "CRITICAL"),
                    )
                )

            # corpus_poisoning: SUSPICIOUS_CHUNK or RETRIEVAL_ANOMALY
            elif result.failure_type in (
                FailureType.SUSPICIOUS_CHUNK,
                FailureType.RETRIEVAL_ANOMALY,
            ):
                failures.append(
                    StageFailure(
                        stage="SECURITY",
                        failure_mode="corpus_poisoning",
                        evidence=result.evidence,
                        severity=SEVERITY_MAP.get(result.failure_type, "HIGH"),
                    )
                )

        return failures

    def _build_failure_chain(self, stage_failures: list[StageFailure]) -> list[str]:
        """Build ordered failure chain showing how failures propagated."""
        # Sort failures by stage order
        sorted_failures = sorted(
            stage_failures, key=lambda f: STAGE_ORDER.get(f.stage, 999)
        )

        # Build chain strings: "STAGE → failure_mode"
        chain = [f"{f.stage} → {f.failure_mode}" for f in sorted_failures]

        return chain

    def _identify_primary_stage(self, stage_failures: list[StageFailure]) -> str:
        """Identify the primary (earliest) failure stage."""
        if not stage_failures:
            return "UNKNOWN"

        # Sort by stage order and return the earliest
        sorted_failures = sorted(
            stage_failures, key=lambda f: STAGE_ORDER.get(f.stage, 999)
        )

        return sorted_failures[0].stage
