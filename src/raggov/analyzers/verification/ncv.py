"""Node-wise Consistency Verification for RAG pipelines."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.retrieval.inconsistency import has_suspicious_negation_pair, terms
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.taxonomy import DEFAULT_REMEDIATIONS


class NCVNode(str, Enum):
    """Discrete RAG pipeline nodes used by NCV."""

    QUERY_UNDERSTANDING = "QUERY_UNDERSTANDING"
    RETRIEVAL_QUALITY = "RETRIEVAL_QUALITY"
    CONTEXT_ASSEMBLY = "CONTEXT_ASSEMBLY"
    CLAIM_GROUNDING = "CLAIM_GROUNDING"
    ANSWER_COMPLETENESS = "ANSWER_COMPLETENESS"


@dataclass
class NodeResult:
    """Binary/ternary verification result for one NCV node."""

    node: NCVNode
    status: Literal["pass", "fail", "uncertain"]
    score: float | None
    note: str


@dataclass
class NCVReport:
    """Serialized NCV verification report."""

    node_results: list[NodeResult]
    first_failing_node: NCVNode | None
    pipeline_health_score: float
    bottleneck_description: str


REMEDIATIONS = {
    NCVNode.QUERY_UNDERSTANDING: DEFAULT_REMEDIATIONS[FailureType.SCOPE_VIOLATION],
    NCVNode.RETRIEVAL_QUALITY: DEFAULT_REMEDIATIONS[FailureType.INSUFFICIENT_CONTEXT],
    NCVNode.CONTEXT_ASSEMBLY: DEFAULT_REMEDIATIONS[FailureType.INCONSISTENT_CHUNKS],
    NCVNode.CLAIM_GROUNDING: DEFAULT_REMEDIATIONS[FailureType.UNSUPPORTED_CLAIM],
    NCVNode.ANSWER_COMPLETENESS: "Answer omitted the expected answer type for the query. Review generation constraints or answer formatting.",
}

FAILURE_MAP = {
    NCVNode.QUERY_UNDERSTANDING: (FailureType.SCOPE_VIOLATION, FailureStage.RETRIEVAL),
    NCVNode.RETRIEVAL_QUALITY: (FailureType.INSUFFICIENT_CONTEXT, FailureStage.RETRIEVAL),
    NCVNode.CONTEXT_ASSEMBLY: (FailureType.INCONSISTENT_CHUNKS, FailureStage.RETRIEVAL),
    NCVNode.CLAIM_GROUNDING: (FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
    NCVNode.ANSWER_COMPLETENESS: (FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
}


class NCVPipelineVerifier(BaseAnalyzer):
    """Run cheap node-wise consistency verification across the RAG pipeline."""

    weight = 0.6

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        fail_fast = bool(self.config.get("fail_fast", True))
        node_results: list[NodeResult] = []

        query_result = self._check_query_understanding(run)
        node_results.append(query_result)

        # Product exception: with empty chunks, record the first two failures even in fail-fast mode.
        if not run.retrieved_chunks:
            retrieval_result = self._force_empty_retrieval_quality_failure()
            node_results.append(retrieval_result)
            return self._build_result(node_results)

        if fail_fast and query_result.status == "fail":
            return self._build_result(node_results)

        retrieval_result = self._check_retrieval_quality(run)
        node_results.append(retrieval_result)
        if fail_fast and retrieval_result.status == "fail":
            return self._build_result(node_results)

        context_result = self._check_context_assembly(run)
        node_results.append(context_result)
        if fail_fast and context_result.status == "fail":
            return self._build_result(node_results)

        grounding_result = self._check_claim_grounding(run)
        node_results.append(grounding_result)
        if fail_fast and grounding_result.status == "fail":
            return self._build_result(node_results)

        completeness_result = self._check_answer_completeness(run)
        node_results.append(completeness_result)

        return self._build_result(node_results)

    def _check_query_understanding(self, run: RAGRun) -> NodeResult:
        scope = ScopeViolationAnalyzer()
        query_terms = scope._terms(run.query)
        if not query_terms:
            return NodeResult(
                node=NCVNode.QUERY_UNDERSTANDING,
                status="uncertain",
                score=None,
                note="no meaningful query terms available",
            )

        if not run.retrieved_chunks:
            return NodeResult(
                node=NCVNode.QUERY_UNDERSTANDING,
                status="fail",
                score=0.0,
                note="no retrieved chunks available to match query terms",
            )

        all_chunk_terms: set[str] = set()
        for chunk in run.retrieved_chunks:
            all_chunk_terms |= scope._terms(chunk.text)

        overlap = len(query_terms & all_chunk_terms) / len(query_terms)
        status: Literal["pass", "fail", "uncertain"] = "fail" if overlap < 0.2 else "pass"
        note = (
            f"query-term overlap {overlap:.2f} is below 0.20"
            if status == "fail"
            else f"query-term overlap {overlap:.2f} is acceptable"
        )
        return NodeResult(
            node=NCVNode.QUERY_UNDERSTANDING,
            status=status,
            score=round(overlap, 2),
            note=note,
        )

    def _check_retrieval_quality(self, run: RAGRun) -> NodeResult:
        scores = [chunk.score for chunk in run.retrieved_chunks if chunk.score is not None]
        threshold = float(self.config.get("score_threshold_retrieval", 0.5))

        if not scores:
            return NodeResult(
                node=NCVNode.RETRIEVAL_QUALITY,
                status="uncertain",
                score=None,
                note="no retrieval scores available",
            )

        mean_score = statistics.mean(scores)
        variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
        max_score = max(scores)

        if mean_score < threshold:
            return NodeResult(
                node=NCVNode.RETRIEVAL_QUALITY,
                status="fail",
                score=round(mean_score, 2),
                note=f"mean retrieval score {mean_score:.2f} is below {threshold:.2f}",
            )
        if variance > 0.15 and max_score > 0.85:
            return NodeResult(
                node=NCVNode.RETRIEVAL_QUALITY,
                status="uncertain",
                score=round(mean_score, 2),
                note=f"score variance {variance:.2f} with max score {max_score:.2f} suggests an outlier-heavy retrieval set",
            )
        return NodeResult(
            node=NCVNode.RETRIEVAL_QUALITY,
            status="pass",
            score=round(mean_score, 2),
            note=f"mean retrieval score {mean_score:.2f} is acceptable",
        )

    def _force_empty_retrieval_quality_failure(self) -> NodeResult:
        return NodeResult(
            node=NCVNode.RETRIEVAL_QUALITY,
            status="fail",
            score=0.0,
            note="no retrieved chunks available to score retrieval quality",
        )

    def _check_context_assembly(self, run: RAGRun) -> NodeResult:
        duplicate_score = 0.0
        duplicate_pair: tuple[str, str] | None = None
        has_negation_conflict = False

        for index, left in enumerate(run.retrieved_chunks):
            for right in run.retrieved_chunks[index + 1 :]:
                if has_suspicious_negation_pair(left, right):
                    has_negation_conflict = True

                left_terms = terms(left.text)
                right_terms = terms(right.text)
                union = left_terms | right_terms
                similarity = len(left_terms & right_terms) / len(union) if union else 0.0
                if similarity > duplicate_score:
                    duplicate_score = similarity
                    duplicate_pair = (left.chunk_id, right.chunk_id)

        if duplicate_pair is not None and duplicate_score > 0.85:
            return NodeResult(
                node=NCVNode.CONTEXT_ASSEMBLY,
                status="fail",
                score=round(duplicate_score, 2),
                note=f"duplicate chunks {duplicate_pair[0]} and {duplicate_pair[1]} share Jaccard {duplicate_score:.2f}",
            )
        if has_negation_conflict:
            return NodeResult(
                node=NCVNode.CONTEXT_ASSEMBLY,
                status="uncertain",
                score=None,
                note="retrieved chunks contain a possible negation contradiction",
            )
        return NodeResult(
            node=NCVNode.CONTEXT_ASSEMBLY,
            status="pass",
            score=None,
            note="context assembly is coherent",
        )

    def _check_claim_grounding(self, run: RAGRun) -> NodeResult:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", run.final_answer) if segment.strip()]
        if not sentences:
            sentences = [run.final_answer.strip()] if run.final_answer.strip() else []
        if not sentences:
            return NodeResult(
                node=NCVNode.CLAIM_GROUNDING,
                status="uncertain",
                score=None,
                note="no answer sentences available for grounding",
            )

        grounding_helper = ClaimGroundingAnalyzer()
        grounded = 0
        meaningful_sentences = 0

        for sentence in sentences:
            sentence_terms = terms(sentence)
            if not sentence_terms:
                continue
            meaningful_sentences += 1
            best_ratio = 0.0
            for chunk in run.retrieved_chunks:
                chunk_terms = terms(chunk.text)
                overlap_ratio = len(sentence_terms & chunk_terms) / len(sentence_terms)
                best_ratio = max(best_ratio, overlap_ratio)
            if best_ratio >= 0.4:
                grounded += 1

        if meaningful_sentences == 0:
            return NodeResult(
                node=NCVNode.CLAIM_GROUNDING,
                status="uncertain",
                score=None,
                note="no meaningful answer terms available for grounding",
            )

        grounded_fraction = grounded / meaningful_sentences
        fail_threshold = float(self.config.get("grounding_threshold", 0.5))
        if grounded_fraction < fail_threshold:
            return NodeResult(
                node=NCVNode.CLAIM_GROUNDING,
                status="fail",
                score=round(grounded_fraction, 2),
                note=f"only {grounded}/{meaningful_sentences} answer sentences are grounded",
            )
        if grounded_fraction < 0.7:
            return NodeResult(
                node=NCVNode.CLAIM_GROUNDING,
                status="uncertain",
                score=round(grounded_fraction, 2),
                note=f"grounded fraction {grounded_fraction:.2f} is borderline",
            )
        return NodeResult(
            node=NCVNode.CLAIM_GROUNDING,
            status="pass",
            score=round(grounded_fraction, 2),
            note=f"grounded fraction {grounded_fraction:.2f} is strong",
        )

    def _check_answer_completeness(self, run: RAGRun) -> NodeResult:
        query = run.query.lower()
        answer = run.final_answer.strip()
        answer_lower = answer.lower()

        question_type: str | None = None
        if any(phrase in query for phrase in ("how many", "what number", "count")):
            question_type = "number"
            present = bool(re.search(r"\b\d+\b", answer))
        elif any(phrase in query for phrase in ("when", "what date", "what year")):
            question_type = "date"
            present = bool(
                re.search(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b", answer)
                or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", answer_lower)
            )
        elif any(phrase in query for phrase in ("who", "whose")):
            question_type = "entity"
            present = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer))
        elif "where" in query:
            question_type = "location"
            present = bool(
                re.search(r"\b(?:in|at|from|near)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)
                or re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)
            )
        elif any(phrase in query for phrase in ("yes or no", "is it", "does it")):
            question_type = "yes_no"
            present = answer_lower.startswith("yes") or answer_lower.startswith("no")
        else:
            return NodeResult(
                node=NCVNode.ANSWER_COMPLETENESS,
                status="uncertain",
                score=None,
                note="query does not match a supported answer-type heuristic",
            )

        if present:
            return NodeResult(
                node=NCVNode.ANSWER_COMPLETENESS,
                status="pass",
                score=None,
                note=f"answer includes the expected {question_type} signal",
            )
        return NodeResult(
            node=NCVNode.ANSWER_COMPLETENESS,
            status="fail",
            score=None,
            note=f"answer is missing the expected {question_type} signal",
        )

    def _build_result(self, node_results: list[NodeResult]) -> AnalyzerResult:
        first_failing_node = next(
            (node_result.node for node_result in node_results if node_result.status == "fail"),
            None,
        )
        passing_nodes = sum(1 for node_result in node_results if node_result.status == "pass")
        pipeline_health_score = passing_nodes / len(node_results) if node_results else 0.0

        if first_failing_node is not None:
            bottleneck_description = (
                f"Pipeline fails at {first_failing_node.value}: "
                f"{next(node.note for node in node_results if node.node == first_failing_node)}"
            )
        else:
            first_uncertain = next(
                (node_result for node_result in node_results if node_result.status == "uncertain"),
                None,
            )
            if first_uncertain is not None:
                bottleneck_description = (
                    f"Pipeline is uncertain at {first_uncertain.node.value}: {first_uncertain.note}"
                )
            else:
                bottleneck_description = "Pipeline is healthy across all evaluated NCV nodes"

        report = NCVReport(
            node_results=node_results,
            first_failing_node=first_failing_node,
            pipeline_health_score=round(pipeline_health_score, 2),
            bottleneck_description=bottleneck_description,
        )

        evidence = [
            json.dumps(
                {
                    "node_results": [asdict(node_result) for node_result in report.node_results],
                    "first_failing_node": (
                        report.first_failing_node.value if report.first_failing_node is not None else None
                    ),
                    "pipeline_health_score": report.pipeline_health_score,
                    "bottleneck_description": report.bottleneck_description,
                }
            ),
            report.bottleneck_description,
        ]

        if report.first_failing_node is not None:
            failure_type, stage = FAILURE_MAP[report.first_failing_node]
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=failure_type,
                stage=stage,
                score=report.pipeline_health_score,
                evidence=evidence,
                remediation=REMEDIATIONS.get(report.first_failing_node, DEFAULT_REMEDIATIONS[failure_type]),
            )

        if any(node_result.status == "uncertain" for node_result in node_results):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                score=report.pipeline_health_score,
                evidence=evidence,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            score=report.pipeline_health_score,
            evidence=evidence,
        )
