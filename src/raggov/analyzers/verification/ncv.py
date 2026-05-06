"""Node-wise evidence aggregation verifier inspired by NCV-style verification.

This analyzer upgrades the legacy NCVPipelineVerifier into a node-wise evidence
aggregation analyzer over existing GovRAG reports. The implementation is a
practical architecture upgrade with explicit heuristic fallbacks; it is not a
research-faithful NCV implementation and it is not a RAGChecker, RAGAS,
DeepEval, RefChecker, Layer6, or A2P implementation. It is not recommended for
production gating until calibrated on labeled pipeline traces.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.inconsistency import has_suspicious_negation_pair, terms
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.models.citation_faithfulness import CitationSupportLabel
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType, SecurityRisk
from raggov.models.ncv import (
    NCVCalibrationStatus,
    NCVEvidenceSignal,
    NCVMethodType,
    NCVNode,
    NCVNodeResult,
    NCVNodeStatus,
    NCVReport,
)
from raggov.models.retrieval_diagnosis import RetrievalFailureType
from raggov.models.retrieval_evidence import EvidenceRole, QueryRelevanceLabel
from raggov.models.result_index import AnalyzerResultIndex
from raggov.models.run import RAGRun
from raggov.models.version_validity import DocumentValidityStatus
from raggov.parser_validation.models import ParserFailureType, ParserSeverity
from raggov.taxonomy import DEFAULT_REMEDIATIONS


NODE_ORDER = [
    NCVNode.QUERY_UNDERSTANDING,
    NCVNode.PARSER_VALIDITY,
    NCVNode.RETRIEVAL_COVERAGE,
    NCVNode.RETRIEVAL_PRECISION,
    NCVNode.CONTEXT_ASSEMBLY,
    NCVNode.VERSION_VALIDITY,
    NCVNode.CLAIM_SUPPORT,
    NCVNode.CITATION_SUPPORT,
    NCVNode.ANSWER_COMPLETENESS,
    NCVNode.SECURITY_RISK,
]

FAILURE_MAP = {
    NCVNode.QUERY_UNDERSTANDING: (FailureType.SCOPE_VIOLATION, FailureStage.RETRIEVAL),
    NCVNode.PARSER_VALIDITY: (FailureType.PARSER_STRUCTURE_LOSS, FailureStage.PARSING),
    NCVNode.RETRIEVAL_COVERAGE: (FailureType.INSUFFICIENT_CONTEXT, FailureStage.RETRIEVAL),
    NCVNode.RETRIEVAL_PRECISION: (FailureType.RETRIEVAL_ANOMALY, FailureStage.RETRIEVAL),
    NCVNode.CONTEXT_ASSEMBLY: (FailureType.INCONSISTENT_CHUNKS, FailureStage.RETRIEVAL),
    NCVNode.VERSION_VALIDITY: (FailureType.STALE_RETRIEVAL, FailureStage.RETRIEVAL),
    NCVNode.CLAIM_SUPPORT: (FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
    NCVNode.CITATION_SUPPORT: (FailureType.CITATION_MISMATCH, FailureStage.GROUNDING),
    NCVNode.ANSWER_COMPLETENESS: (FailureType.GENERATION_IGNORE, FailureStage.GENERATION),
    NCVNode.SECURITY_RISK: (FailureType.SUSPICIOUS_CHUNK, FailureStage.SECURITY),
}

NODE_REMEDIATIONS = {
    NCVNode.QUERY_UNDERSTANDING: DEFAULT_REMEDIATIONS[FailureType.SCOPE_VIOLATION],
    NCVNode.PARSER_VALIDITY: DEFAULT_REMEDIATIONS[FailureType.PARSER_STRUCTURE_LOSS],
    NCVNode.RETRIEVAL_COVERAGE: DEFAULT_REMEDIATIONS[FailureType.INSUFFICIENT_CONTEXT],
    NCVNode.RETRIEVAL_PRECISION: DEFAULT_REMEDIATIONS[FailureType.RETRIEVAL_ANOMALY],
    NCVNode.CONTEXT_ASSEMBLY: DEFAULT_REMEDIATIONS[FailureType.INCONSISTENT_CHUNKS],
    NCVNode.VERSION_VALIDITY: DEFAULT_REMEDIATIONS[FailureType.STALE_RETRIEVAL],
    NCVNode.CLAIM_SUPPORT: DEFAULT_REMEDIATIONS[FailureType.UNSUPPORTED_CLAIM],
    NCVNode.CITATION_SUPPORT: DEFAULT_REMEDIATIONS[FailureType.CITATION_MISMATCH],
    NCVNode.ANSWER_COMPLETENESS: DEFAULT_REMEDIATIONS[FailureType.GENERATION_IGNORE],
    NCVNode.SECURITY_RISK: DEFAULT_REMEDIATIONS[FailureType.SUSPICIOUS_CHUNK],
}


class NCVPipelineVerifier(BaseAnalyzer):
    """Aggregate node-wise evidence across the RAG pipeline.

    This verifier consumes structured prior analyzer reports when available and
    makes every fallback explicit in the NCVReport. It preserves the
    BaseAnalyzer/AnalyzerResult contract and the legacy fail_fast behavior.
    """

    weight = 0.6

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        prior_results = self.config.get("weighted_prior_results") or self.config.get("prior_results", [])
        result_index = AnalyzerResultIndex(prior_results)
        fail_fast = bool(self.config.get("fail_fast", True))
        missing_reports: list[str] = []
        fallback_heuristics_used: list[str] = []
        evidence_reports_used: list[str] = []
        node_results: list[NCVNodeResult] = []

        for node in NODE_ORDER:
            node_result = self._check_node(
                node,
                run,
                result_index,
                missing_reports,
                fallback_heuristics_used,
                evidence_reports_used,
            )
            node_results.append(node_result)

            # Preserve the legacy empty-retrieval behavior by recording query
            # failure and retrieval coverage failure before stopping.
            if (
                fail_fast
                and node_result.status == NCVNodeStatus.FAIL
                and not (node == NCVNode.QUERY_UNDERSTANDING and not run.retrieved_chunks)
            ):
                break

        return self._build_result(
            run=run,
            node_results=node_results,
            evidence_reports_used=self._unique(evidence_reports_used),
            missing_reports=self._unique(missing_reports),
            fallback_heuristics_used=self._unique(fallback_heuristics_used),
        )

    def _check_node(
        self,
        node: NCVNode,
        run: RAGRun,
        result_index: AnalyzerResultIndex,
        missing_reports: list[str],
        fallback_heuristics_used: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        if node == NCVNode.QUERY_UNDERSTANDING:
            return self._check_query_understanding(run, result_index, fallback_heuristics_used, evidence_reports_used)
        if node == NCVNode.PARSER_VALIDITY:
            return self._check_parser_validity(result_index, missing_reports, evidence_reports_used)
        if node == NCVNode.RETRIEVAL_COVERAGE:
            return self._check_retrieval_coverage(run, result_index, missing_reports, evidence_reports_used)
        if node == NCVNode.RETRIEVAL_PRECISION:
            return self._check_retrieval_precision(run, result_index, missing_reports, fallback_heuristics_used, evidence_reports_used)
        if node == NCVNode.CONTEXT_ASSEMBLY:
            return self._check_context_assembly(run, result_index, fallback_heuristics_used, evidence_reports_used)
        if node == NCVNode.VERSION_VALIDITY:
            return self._check_version_validity(run, result_index, missing_reports, evidence_reports_used)
        if node == NCVNode.CLAIM_SUPPORT:
            return self._check_claim_support(run, result_index, fallback_heuristics_used, evidence_reports_used)
        if node == NCVNode.CITATION_SUPPORT:
            return self._check_citation_support(run, result_index, missing_reports, evidence_reports_used)
        if node == NCVNode.ANSWER_COMPLETENESS:
            return self._check_answer_completeness(run)
        return self._check_security_risk(result_index, missing_reports, evidence_reports_used)

    def _check_query_understanding(
        self,
        run: RAGRun,
        result_index: AnalyzerResultIndex,
        fallback_heuristics_used: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        profile = self._get_retrieval_evidence_profile(run, result_index)
        if not run.retrieved_chunks:
            return self._node(
                NCVNode.QUERY_UNDERSTANDING,
                NCVNodeStatus.FAIL,
                "No retrieved chunks are available to evaluate query understanding.",
                signals=[self._signal("no_retrieved_chunks", True, "RAGRun", [], "Retrieval returned no chunks.")],
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.SCOPE_VIOLATION],
            )

        if profile is not None:
            relevance_chunks = [
                chunk for chunk in profile.chunks if chunk.query_relevance_label != QueryRelevanceLabel.UNKNOWN
            ]
            if relevance_chunks:
                relevant = [
                    chunk
                    for chunk in relevance_chunks
                    if chunk.query_relevance_label in {QueryRelevanceLabel.RELEVANT, QueryRelevanceLabel.PARTIAL}
                ]
                evidence_reports_used.append("RetrievalEvidenceProfile")
                status = NCVNodeStatus.PASS if relevant else NCVNodeStatus.WARN
                return self._node(
                    NCVNode.QUERY_UNDERSTANDING,
                    status,
                    (
                        "Retrieval evidence profile has query-relevant chunks."
                        if relevant
                        else "Retrieval evidence profile marks chunks as query-irrelevant."
                    ),
                    signals=[
                        self._signal(
                            "query_relevance_evidence",
                            f"{len(relevant)}/{len(relevance_chunks)}",
                            "RetrievalEvidenceProfile",
                            [chunk.chunk_id for chunk in relevance_chunks],
                            "Structured query relevance labels are available for retrieved chunks.",
                        )
                    ],
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    affected_chunk_ids=[chunk.chunk_id for chunk in relevance_chunks],
                )

        fallback_heuristics_used.append("query_term_overlap")
        scope = ScopeViolationAnalyzer()
        query_terms = scope._terms(run.query)
        if not query_terms:
            return self._node(
                NCVNode.QUERY_UNDERSTANDING,
                NCVNodeStatus.UNCERTAIN,
                "No meaningful query terms are available for lexical fallback.",
                missing_evidence=["query relevance report"],
                fallback_used=True,
                limitations=["Lexical fallback cannot interpret sparse or non-lexical queries."],
            )

        all_chunk_terms: set[str] = set()
        for chunk in run.retrieved_chunks:
            all_chunk_terms |= scope._terms(chunk.text)
        overlap = len(query_terms & all_chunk_terms) / len(query_terms)
        status = NCVNodeStatus.FAIL if overlap < 0.2 else NCVNodeStatus.PASS
        return self._node(
            NCVNode.QUERY_UNDERSTANDING,
            status,
            f"Lexical query-term overlap is {overlap:.2f}.",
            score=round(overlap, 2),
            signals=[
                self._signal(
                    "query_term_overlap",
                    round(overlap, 2),
                    "heuristic_fallback",
                    [],
                    "Query terms were compared against retrieved chunk terms.",
                    "Lexical overlap is only a fallback and can miss paraphrase or synonym relevance.",
                )
            ],
            recommended_fix=DEFAULT_REMEDIATIONS[FailureType.SCOPE_VIOLATION] if status == NCVNodeStatus.FAIL else None,
            method_type=NCVMethodType.HEURISTIC_BASELINE,
            fallback_used=True,
            limitations=["Lexical query overlap is not semantic query understanding."],
        )

    def _check_parser_validity(
        self,
        result_index: AnalyzerResultIndex,
        missing_reports: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        parser_results = self._get_parser_validation_results(result_index)
        if not parser_results:
            missing_reports.append("parser_validation_results")
            return self._skip(NCVNode.PARSER_VALIDITY, "Parser validation results are unavailable.")

        evidence_reports_used.append("ParserValidationResults")
        failing_types = {
            ParserFailureType.TABLE_STRUCTURE_LOSS.value,
            ParserFailureType.HIERARCHY_FLATTENING.value,
            ParserFailureType.METADATA_LOSS.value,
            ParserFailureType.PROVENANCE_MISSING.value,
            ParserFailureType.CHUNK_BOUNDARY_DAMAGE.value,
        }
        signals: list[NCVEvidenceSignal] = []
        fail_findings: list[Any] = []
        warn_findings: list[Any] = []
        for finding in parser_results:
            # Handle both ParserFinding (from diagnostic_rollup) and AnalyzerResult
            severity = self._enum_value(getattr(finding, "severity", getattr(finding, "status", ""))).upper()
            failure_type = self._enum_value(getattr(finding, "failure_type", ""))
            
            # If failure_type is generic or missing, refine it from evidence
            if failure_type in {"None", "", "METADATA_LOSS"} or not failure_type:
                evidence = getattr(finding, "evidence", [])
                if any("parser_validation_profile_missing" in str(e) for e in evidence):
                    failure_type = "PARSER_PROFILE_MISSING"
                elif any("parser_validation_profile_malformed" in str(e) for e in evidence):
                    failure_type = "PARSER_PROFILE_MALFORMED"

            signal = self._signal(
                "parser_validation_finding",
                failure_type,
                "ParserValidationResults",
                self._parser_evidence_ids(finding),
                f"Parser validation reported {failure_type}.",
            )
            
            # Match normalized severity or check against specific failing types
            is_fail = severity == "FAIL" or (failure_type in failing_types and severity == ParserSeverity.FAIL.value)
            is_warn = severity == "WARN" or (failure_type in failing_types and severity == ParserSeverity.WARN.value)
            
            if is_fail:
                fail_findings.append(finding)
                signals.append(signal)
            elif is_warn:
                warn_findings.append(finding)
                signals.append(signal)

        if fail_findings:
            return self._node(
                NCVNode.PARSER_VALIDITY,
                NCVNodeStatus.FAIL,
                "Parser validation found blocking structural/provenance damage.",
                signals=signals,
                recommended_fix=getattr(fail_findings[0], "remediation", None) or DEFAULT_REMEDIATIONS[FailureType.PARSER_STRUCTURE_LOSS],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                affected_chunk_ids=self._parser_finding_chunk_ids(fail_findings),
                alternative_explanations=list(getattr(fail_findings[0], "alternative_explanations", ()) or ()),
            )
        if warn_findings:
            return self._node(
                NCVNode.PARSER_VALIDITY,
                NCVNodeStatus.WARN,
                "Parser validation found non-blocking structural/provenance warnings.",
                signals=signals,
                recommended_fix=getattr(warn_findings[0], "remediation", None),
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                affected_chunk_ids=self._parser_finding_chunk_ids(warn_findings),
            )
        return self._node(
            NCVNode.PARSER_VALIDITY,
            NCVNodeStatus.PASS,
            "Parser validation results contain no NCV-blocking findings.",
            signals=[self._signal("parser_validation_available", True, "ParserValidationResults", [], "Parser validation results were available.")],
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        )

    def _check_retrieval_coverage(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        missing_reports: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        if not run.retrieved_chunks:
            return self._node(
                NCVNode.RETRIEVAL_COVERAGE,
                NCVNodeStatus.FAIL,
                "No retrieved chunks are available for retrieval coverage.",
                signals=[self._signal("no_retrieved_chunks", True, "RAGRun", [], "Retrieval returned no chunks for coverage.")],
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.INSUFFICIENT_CONTEXT],
            )

        diagnosis = self._get_retrieval_diagnosis_report(run, prior_results)
        if diagnosis is not None:
            evidence_reports_used.append("RetrievalDiagnosisReport")
            primary = self._enum_value(diagnosis.primary_failure_type)
            if diagnosis.primary_failure_type == RetrievalFailureType.RETRIEVAL_MISS:
                return self._node(
                    NCVNode.RETRIEVAL_COVERAGE,
                    NCVNodeStatus.FAIL,
                    "Retrieval diagnosis reports a retrieval miss.",
                    signals=[
                        self._signal(
                            "retrieval_primary_failure",
                            primary,
                            "RetrievalDiagnosisReport",
                            diagnosis.affected_claim_ids,
                            "Primary retrieval diagnosis is retrieval_miss.",
                        )
                    ],
                    missing_evidence=list(diagnosis.missing_reports),
                    affected_claim_ids=list(diagnosis.affected_claim_ids),
                    affected_chunk_ids=list(diagnosis.candidate_chunk_ids),
                    affected_doc_ids=list(diagnosis.invalid_retrieved_doc_ids),
                    recommended_fix=diagnosis.recommended_fix,
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    limitations=list(diagnosis.limitations),
                )
            if diagnosis.primary_failure_type == RetrievalFailureType.INSUFFICIENT_EVIDENCE_TO_DIAGNOSE:
                return self._node(
                    NCVNode.RETRIEVAL_COVERAGE,
                    NCVNodeStatus.UNCERTAIN,
                    "Retrieval diagnosis lacks enough upstream evidence to assess coverage.",
                    signals=[
                        self._signal(
                            "retrieval_primary_failure",
                            primary,
                            "RetrievalDiagnosisReport",
                            [],
                            "Structured retrieval diagnosis could not resolve coverage.",
                        )
                    ],
                    missing_evidence=list(diagnosis.missing_reports),
                    recommended_fix=diagnosis.recommended_fix,
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    limitations=list(diagnosis.limitations),
                )
            return self._node(
                NCVNode.RETRIEVAL_COVERAGE,
                NCVNodeStatus.PASS,
                f"Retrieval diagnosis primary failure is {primary}.",
                signals=[
                    self._signal(
                        "retrieval_primary_failure",
                        primary,
                        "RetrievalDiagnosisReport",
                        diagnosis.affected_claim_ids,
                        "Structured retrieval diagnosis does not indicate retrieval miss.",
                    )
                ],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        sufficiency = self._get_sufficiency_result(prior_results)
        grounding = self._get_claim_grounding_result(prior_results)
        if sufficiency is not None and grounding is not None and not sufficiency.sufficient and self._unsupported_claims(grounding):
            evidence_reports_used.extend(["SufficiencyResult", "ClaimGroundingAnalyzer"])
            return self._node(
                NCVNode.RETRIEVAL_COVERAGE,
                NCVNodeStatus.FAIL,
                "Sufficiency is insufficient and claim grounding has unsupported claims.",
                signals=[
                    self._signal("sufficiency_label", sufficiency.sufficiency_label, "SufficiencyResult", sufficiency.affected_claims, "Structured sufficiency result says evidence is insufficient."),
                    self._signal("unsupported_claims_present", True, "ClaimGroundingAnalyzer", self._claim_ids(grounding), "Claim grounding reports unsupported claims."),
                ],
                missing_evidence=list(sufficiency.missing_evidence),
                affected_claim_ids=list(sufficiency.affected_claims) or self._claim_ids(grounding),
                affected_chunk_ids=self._affected_chunks_from_grounding(grounding),
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.INSUFFICIENT_CONTEXT],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                limitations=list(sufficiency.limitations),
            )

        missing_reports.append("retrieval_diagnosis_report")
        return self._node(
            NCVNode.RETRIEVAL_COVERAGE,
            NCVNodeStatus.UNCERTAIN,
            "Required retrieval coverage reports are unavailable.",
            missing_evidence=["retrieval_diagnosis_report"],
            method_type=NCVMethodType.PRACTICAL_APPROXIMATION,
            limitations=["Coverage is not inferred from retrieval score heuristics."],
        )

    def _check_retrieval_precision(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        missing_reports: list[str],
        fallback_heuristics_used: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        diagnosis = self._get_retrieval_diagnosis_report(run, prior_results)
        profile = self._get_retrieval_evidence_profile(run, prior_results)
        if diagnosis is not None:
            evidence_reports_used.append("RetrievalDiagnosisReport")
            primary = diagnosis.primary_failure_type
            if primary == RetrievalFailureType.RETRIEVAL_NOISE:
                status = NCVNodeStatus.FAIL if len(diagnosis.noisy_chunk_ids) >= max(2, len(run.retrieved_chunks)) else NCVNodeStatus.WARN
                return self._node(
                    NCVNode.RETRIEVAL_PRECISION,
                    status,
                    "Retrieval diagnosis reports retrieval noise.",
                    signals=[self._signal("retrieval_primary_failure", primary.value, "RetrievalDiagnosisReport", diagnosis.noisy_chunk_ids, "Structured retrieval diagnosis found noisy chunks.")],
                    affected_chunk_ids=list(diagnosis.noisy_chunk_ids),
                    recommended_fix=diagnosis.recommended_fix,
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    limitations=list(diagnosis.limitations),
                )
            if primary == RetrievalFailureType.RANK_FAILURE_UNKNOWN:
                return self._node(
                    NCVNode.RETRIEVAL_PRECISION,
                    NCVNodeStatus.WARN,
                    "Retrieval diagnosis reports rank failure unknown.",
                    signals=[self._signal("retrieval_primary_failure", primary.value, "RetrievalDiagnosisReport", diagnosis.candidate_chunk_ids, "Structured retrieval diagnosis found unresolved ranking evidence.")],
                    affected_chunk_ids=list(diagnosis.candidate_chunk_ids),
                    recommended_fix=diagnosis.recommended_fix,
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    limitations=list(diagnosis.limitations),
                )

        if profile is not None and (profile.noisy_chunk_ids or any(chunk.evidence_role == EvidenceRole.NOISE for chunk in profile.chunks)):
            evidence_reports_used.append("RetrievalEvidenceProfile")
            noisy_ids = self._unique(list(profile.noisy_chunk_ids) + [chunk.chunk_id for chunk in profile.chunks if chunk.evidence_role == EvidenceRole.NOISE])
            return self._node(
                NCVNode.RETRIEVAL_PRECISION,
                NCVNodeStatus.WARN,
                "Retrieval evidence profile marks chunks as noise.",
                signals=[self._signal("noisy_chunk_ids", len(noisy_ids), "RetrievalEvidenceProfile", noisy_ids, "Structured profile identifies retrieval noise.")],
                affected_chunk_ids=noisy_ids,
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        if self.config.get("allow_retrieval_precision_fallback", False) or self.config.get("score_threshold_retrieval") is not None:
            scores = [chunk.score for chunk in run.retrieved_chunks if chunk.score is not None]
            threshold = float(self.config.get("score_threshold_retrieval", 0.5))
            fallback_heuristics_used.append("mean_retrieval_score_threshold")
            if not scores:
                return self._node(
                    NCVNode.RETRIEVAL_PRECISION,
                    NCVNodeStatus.UNCERTAIN,
                    "No retrieval scores are available for precision fallback.",
                    missing_evidence=["retrieval_diagnosis_report", "retrieval_evidence_profile", "retrieval scores"],
                    method_type=NCVMethodType.HEURISTIC_BASELINE,
                    fallback_used=True,
                )
            mean_score = sum(scores) / len(scores)
            if mean_score < threshold:
                return self._node(
                    NCVNode.RETRIEVAL_PRECISION,
                    NCVNodeStatus.FAIL,
                    f"Mean retrieval score {mean_score:.2f} is below {threshold:.2f}.",
                    score=round(mean_score, 2),
                    signals=[self._signal("mean_retrieval_score", round(mean_score, 2), "heuristic_fallback", [], "Legacy mean-score threshold was used as a precision fallback.", "Retrieval scores are model-specific and uncalibrated.")],
                    recommended_fix=DEFAULT_REMEDIATIONS[FailureType.RETRIEVAL_ANOMALY],
                    method_type=NCVMethodType.HEURISTIC_BASELINE,
                    fallback_used=True,
                    limitations=["Mean retrieval score threshold is a legacy heuristic, not calibrated precision."],
                )

        if diagnosis is None and profile is None:
            missing_reports.extend(["retrieval_diagnosis_report", "retrieval_evidence_profile"])
        return self._node(
            NCVNode.RETRIEVAL_PRECISION,
            NCVNodeStatus.PASS,
            "No retrieval precision warning is available from structured reports.",
            missing_evidence=[] if diagnosis is not None or profile is not None else ["retrieval_diagnosis_report", "retrieval_evidence_profile"],
            method_type=NCVMethodType.PRACTICAL_APPROXIMATION,
            limitations=["Precision is only considered healthy when no structured noise/rank warning is present."],
        )

    def _check_context_assembly(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        fallback_heuristics_used: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        profile = self._get_retrieval_evidence_profile(run, prior_results)
        if profile is not None and profile.contradictory_pairs:
            evidence_reports_used.append("RetrievalEvidenceProfile")
            pair_ids = [item for pair in profile.contradictory_pairs for item in pair]
            return self._node(
                NCVNode.CONTEXT_ASSEMBLY,
                NCVNodeStatus.FAIL,
                "Retrieval evidence profile reports contradictory context pairs.",
                signals=[self._signal("contradictory_pairs", len(profile.contradictory_pairs), "RetrievalEvidenceProfile", pair_ids, "Structured profile reports context contradictions.")],
                affected_chunk_ids=self._unique(pair_ids),
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.INCONSISTENT_CHUNKS],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        inconsistency = self._get_result_by_failure_type(prior_results, FailureType.INCONSISTENT_CHUNKS)
        if inconsistency is not None and inconsistency.status in {"fail", "warn"}:
            evidence_reports_used.append(inconsistency.analyzer_name)
            return self._node(
                NCVNode.CONTEXT_ASSEMBLY,
                NCVNodeStatus.FAIL if inconsistency.status == "fail" else NCVNodeStatus.WARN,
                "Prior inconsistency analyzer reported context assembly risk.",
                signals=[self._signal("inconsistent_chunks_result", inconsistency.status, inconsistency.analyzer_name, [], "Prior analyzer reported inconsistent retrieved chunks.")],
                recommended_fix=inconsistency.remediation or DEFAULT_REMEDIATIONS[FailureType.INCONSISTENT_CHUNKS],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        fallback_heuristics_used.append("jaccard_duplicate_or_negation_pair")
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
            return self._node(
                NCVNode.CONTEXT_ASSEMBLY,
                NCVNodeStatus.FAIL,
                f"Legacy Jaccard fallback found duplicate chunks at {duplicate_score:.2f}.",
                score=round(duplicate_score, 2),
                signals=[self._signal("duplicate_chunk_jaccard", round(duplicate_score, 2), "heuristic_fallback", list(duplicate_pair), "Legacy duplicate detection found near-identical chunks.")],
                affected_chunk_ids=list(duplicate_pair),
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.INCONSISTENT_CHUNKS],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
                limitations=["Jaccard duplicate detection is a lexical fallback."],
            )
        if has_negation_conflict:
            return self._node(
                NCVNode.CONTEXT_ASSEMBLY,
                NCVNodeStatus.WARN,
                "Legacy negation-pair fallback found a possible contradiction.",
                signals=[self._signal("negation_pair_detected", True, "heuristic_fallback", [], "Legacy negation-pair detection found possible contradiction.")],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
                limitations=["Negation-pair detection is not a contradiction verifier."],
            )
        return self._node(
            NCVNode.CONTEXT_ASSEMBLY,
            NCVNodeStatus.PASS,
            "No context assembly contradiction or duplicate signal was found.",
            method_type=NCVMethodType.HEURISTIC_BASELINE,
            fallback_used=True,
            limitations=["Context assembly pass is based on limited legacy fallback when structured contradiction reports are absent."],
        )

    def _check_version_validity(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        missing_reports: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        report = self._get_version_validity_report(run, prior_results)
        if report is None:
            missing_reports.append("version_validity_report")
            return self._skip(NCVNode.VERSION_VALIDITY, "Version validity report is unavailable.")

        evidence_reports_used.append("VersionValidityReport")
        invalid_cited = self._unique(report.expired_doc_ids + report.superseded_doc_ids + report.withdrawn_doc_ids + report.not_yet_effective_doc_ids + report.deprecated_doc_ids)
        invalid_cited = [doc_id for doc_id in invalid_cited if doc_id in set(run.cited_doc_ids or invalid_cited)]
        invalid_retrieved = self._unique(report.expired_doc_ids + report.superseded_doc_ids + report.withdrawn_doc_ids + report.not_yet_effective_doc_ids + report.deprecated_doc_ids)
        warning_docs = self._unique(report.metadata_missing_doc_ids + report.amended_doc_ids + [record.doc_id for record in report.document_records if record.validity_status in {DocumentValidityStatus.APPLICABILITY_UNKNOWN, DocumentValidityStatus.UNKNOWN}])
        if invalid_cited:
            return self._node(
                NCVNode.VERSION_VALIDITY,
                NCVNodeStatus.FAIL,
                "Version validity report found invalid cited documents.",
                signals=[self._signal("invalid_cited_doc_ids", len(invalid_cited), "VersionValidityReport", invalid_cited, "Cited documents are expired, superseded, withdrawn, deprecated, or not yet effective.")],
                affected_doc_ids=invalid_cited,
                affected_claim_ids=list(report.high_risk_claim_ids),
                recommended_fix=DEFAULT_REMEDIATIONS[FailureType.STALE_RETRIEVAL],
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                limitations=list(report.limitations),
            )
        if invalid_retrieved or warning_docs:
            docs = self._unique(invalid_retrieved + warning_docs)
            return self._node(
                NCVNode.VERSION_VALIDITY,
                NCVNodeStatus.WARN,
                "Version validity report found invalid retrieved documents or uncertain metadata.",
                signals=[self._signal("version_validity_warning_docs", len(docs), "VersionValidityReport", docs, "Retrieved documents include invalid, amended, metadata-missing, or unknown-validity sources.")],
                affected_doc_ids=docs,
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                limitations=list(report.limitations),
            )
        return self._node(
            NCVNode.VERSION_VALIDITY,
            NCVNodeStatus.PASS,
            "Version validity report found no invalid or uncertain source validity signal.",
            signals=[self._signal("version_validity_report_available", True, "VersionValidityReport", [], "Version validity report was available.")],
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        )

    def _check_claim_support(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        fallback_heuristics_used: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        grounding = self._get_claim_grounding_result(prior_results)
        if grounding is not None:
            evidence_reports_used.append("ClaimGroundingAnalyzer")
            unsupported = [claim for claim in (grounding.claim_results or []) if claim.label in {"unsupported", "contradicted"}]

            # Surface external claim verifier provider if present
            external_verifier_signals: list[NCVEvidenceSignal] = []
            verifier_provider = self._get_claim_verifier_provider(grounding)
            if verifier_provider:
                external_verifier_signals.append(
                    self._signal(
                        "external_claim_verifier_provider",
                        verifier_provider,
                        "ClaimGroundingAnalyzer",
                        [],
                        f"Claim verification was performed by external provider: {verifier_provider}.",
                    )
                )

            if unsupported:
                labels = self._unique([claim.label for claim in unsupported])
                failure_type = FailureType.CONTRADICTED_CLAIM if "contradicted" in labels else FailureType.UNSUPPORTED_CLAIM
                return self._node(
                    NCVNode.CLAIM_SUPPORT,
                    NCVNodeStatus.FAIL,
                    f"Claim grounding reports {', '.join(labels)} claims.",
                    signals=[
                        self._signal(
                            "claim_grounding_labels",
                            ",".join(labels),
                            "ClaimGroundingAnalyzer",
                            self._claim_ids(grounding),
                            "Structured claim grounding reports unsupported or contradicted claims.",
                        )
                    ] + external_verifier_signals,
                    affected_claim_ids=self._claim_ids(grounding, unsupported_only=True),
                    affected_chunk_ids=self._affected_chunks_from_claim_results(unsupported),
                    recommended_fix=DEFAULT_REMEDIATIONS[failure_type],
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                )
            return self._node(
                NCVNode.CLAIM_SUPPORT,
                NCVNodeStatus.PASS,
                "Claim grounding result has no unsupported or contradicted claims.",
                signals=[
                    self._signal(
                        "claim_grounding_result_available",
                        True,
                        "ClaimGroundingAnalyzer",
                        self._claim_ids(grounding),
                        "Structured claim grounding result was available.",
                    )
                ] + external_verifier_signals,
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        if not self.config.get("allow_claim_support_fallback", True):
            return self._node(
                NCVNode.CLAIM_SUPPORT,
                NCVNodeStatus.SKIP,
                "Claim grounding report is unavailable and claim fallback is disabled.",
                missing_evidence=["ClaimGroundingAnalyzer result"],
                method_type=NCVMethodType.PRACTICAL_APPROXIMATION,
            )

        fallback_heuristics_used.append("answer_sentence_token_overlap")
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", run.final_answer) if segment.strip()]
        if not sentences and run.final_answer.strip():
            sentences = [run.final_answer.strip()]
        if not sentences:
            return self._node(
                NCVNode.CLAIM_SUPPORT,
                NCVNodeStatus.UNCERTAIN,
                "No answer sentences are available for grounding fallback.",
                missing_evidence=["answer sentences", "ClaimGroundingAnalyzer result"],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
            )

        grounded = 0
        meaningful = 0
        for sentence in sentences:
            sentence_terms = terms(sentence)
            if not sentence_terms:
                continue
            meaningful += 1
            best_ratio = 0.0
            for chunk in run.retrieved_chunks:
                chunk_terms = terms(chunk.text)
                best_ratio = max(best_ratio, len(sentence_terms & chunk_terms) / len(sentence_terms))
            if best_ratio >= 0.4:
                grounded += 1

        if meaningful == 0:
            return self._node(
                NCVNode.CLAIM_SUPPORT,
                NCVNodeStatus.UNCERTAIN,
                "No meaningful answer terms are available for grounding fallback.",
                missing_evidence=["meaningful answer terms", "ClaimGroundingAnalyzer result"],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
            )
        fraction = grounded / meaningful
        status = NCVNodeStatus.FAIL if fraction < float(self.config.get("grounding_threshold", 0.5)) else NCVNodeStatus.PASS
        if status == NCVNodeStatus.PASS and fraction < 0.7:
            status = NCVNodeStatus.UNCERTAIN
        return self._node(
            NCVNode.CLAIM_SUPPORT,
            status,
            f"Legacy answer sentence token overlap grounded {grounded}/{meaningful} sentences.",
            score=round(fraction, 2),
            signals=[self._signal("answer_sentence_token_overlap_fraction", round(fraction, 2), "heuristic_fallback", [], "Legacy sentence-to-chunk token overlap was used.")],
            recommended_fix=DEFAULT_REMEDIATIONS[FailureType.UNSUPPORTED_CLAIM] if status == NCVNodeStatus.FAIL else None,
            method_type=NCVMethodType.HEURISTIC_BASELINE,
            fallback_used=True,
            limitations=["Sentence token overlap is not claim verification."],
        )

    def _check_citation_support(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
        missing_reports: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        report = self._get_citation_faithfulness_report(run, prior_results)
        probe = self._get_result_by_name(prior_results, "CitationFaithfulnessProbe")
        if report is None and probe is None:
            missing_reports.append("citation_faithfulness_report")
            return self._skip(NCVNode.CITATION_SUPPORT, "Citation faithfulness report is unavailable.")

        if report is not None:
            evidence_reports_used.append("CitationFaithfulnessReport")
            bad_claims = self._unique(report.unsupported_claim_ids + report.contradicted_claim_ids + report.missing_citation_claim_ids)
            bad_docs = list(report.phantom_citation_doc_ids)
            bad_records = [
                record
                for record in report.records
                if record.citation_support_label
                in {
                    CitationSupportLabel.UNSUPPORTED,
                    CitationSupportLabel.CONTRADICTED,
                    CitationSupportLabel.CITATION_PHANTOM,
                    CitationSupportLabel.CITATION_MISSING,
                }
            ]

            # Surface external citation verifier provider if present
            external_citation_signals: list[NCVEvidenceSignal] = []
            citation_provider = self._get_citation_verifier_provider(report)
            if citation_provider:
                external_citation_signals.append(
                    self._signal(
                        "external_citation_verifier_provider",
                        citation_provider,
                        "CitationFaithfulnessReport",
                        [],
                        f"Citation faithfulness was evaluated by external provider: {citation_provider}.",
                    )
                )

            if bad_claims or bad_docs or bad_records:
                status = NCVNodeStatus.FAIL if bad_docs or report.unsupported_claim_ids or report.contradicted_claim_ids else NCVNodeStatus.WARN
                return self._node(
                    NCVNode.CITATION_SUPPORT,
                    status,
                    "Citation faithfulness report found unsupported, phantom, missing, or mismatched citation support.",
                    signals=[
                        self._signal(
                            "citation_faithfulness_issues",
                            len(bad_claims) + len(bad_docs) + len(bad_records),
                            "CitationFaithfulnessReport",
                            bad_claims + bad_docs,
                            "Structured citation faithfulness report found citation support issues.",
                        )
                    ] + external_citation_signals,
                    affected_claim_ids=bad_claims,
                    affected_doc_ids=bad_docs,
                    affected_chunk_ids=self._unique([chunk_id for record in bad_records for chunk_id in record.cited_chunk_ids]),
                    recommended_fix=DEFAULT_REMEDIATIONS[FailureType.CITATION_MISMATCH],
                    method_type=NCVMethodType.EVIDENCE_AGGREGATION,
                    limitations=list(report.limitations),
                )
            return self._node(
                NCVNode.CITATION_SUPPORT,
                NCVNodeStatus.PASS,
                "Citation faithfulness report found no citation support issue.",
                signals=[
                    self._signal(
                        "citation_faithfulness_report_available",
                        True,
                        "CitationFaithfulnessReport",
                        [],
                        "Citation faithfulness report was available.",
                    )
                ] + external_citation_signals,
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )

        evidence_reports_used.append("CitationFaithfulnessProbe")
        status = NCVNodeStatus.FAIL if probe.status == "fail" else NCVNodeStatus.WARN if probe.status == "warn" else NCVNodeStatus.PASS
        return self._node(
            NCVNode.CITATION_SUPPORT,
            status,
            f"CitationFaithfulnessProbe returned {probe.status}.",
            signals=[self._signal("citation_probe_status", probe.status, "CitationFaithfulnessProbe", [], "CitationFaithfulnessProbe prior result was consumed.")],
            recommended_fix=probe.remediation,
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        )

    def _check_answer_completeness(self, run: RAGRun) -> NCVNodeResult:
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
            present = bool(re.search(r"\b(?:in|at|from|near)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer) or re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer))
        elif any(phrase in query for phrase in ("yes or no", "is it", "does it")):
            question_type = "yes_no"
            present = answer_lower.startswith("yes") or answer_lower.startswith("no")
        else:
            return self._node(
                NCVNode.ANSWER_COMPLETENESS,
                NCVNodeStatus.UNCERTAIN,
                "Query does not match a supported answer-type heuristic.",
                missing_evidence=["supported query type for answer completeness heuristic"],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
                limitations=["Answer completeness currently covers only simple answer-type regex checks."],
            )

        if present:
            return self._node(
                NCVNode.ANSWER_COMPLETENESS,
                NCVNodeStatus.PASS,
                f"Answer includes the expected {question_type} signal.",
                signals=[self._signal("answer_type_present", question_type, "heuristic_baseline", [], "Answer contains the expected answer-type pattern.")],
                method_type=NCVMethodType.HEURISTIC_BASELINE,
                fallback_used=True,
            )
        return self._node(
            NCVNode.ANSWER_COMPLETENESS,
            NCVNodeStatus.FAIL,
            f"Answer is missing the expected {question_type} signal.",
            signals=[self._signal("answer_type_missing", question_type, "heuristic_baseline", [], "Answer lacks the expected answer-type pattern.")],
            recommended_fix="Answer omitted the expected answer type for the query. Review generation constraints or answer formatting.",
            method_type=NCVMethodType.HEURISTIC_BASELINE,
            fallback_used=True,
            limitations=["Answer-type regex is a heuristic baseline and does not prove completeness."],
        )

    def _check_security_risk(
        self,
        prior_results: list[AnalyzerResult],
        missing_reports: list[str],
        evidence_reports_used: list[str],
    ) -> NCVNodeResult:
        security_results = self._get_security_results(prior_results)
        if not security_results:
            missing_reports.append("security_results")
            return self._skip(NCVNode.SECURITY_RISK, "Security analyzer results are unavailable.")

        risky = [result for result in security_results if result.status in {"fail", "warn"}]
        evidence_reports_used.extend(result.analyzer_name for result in security_results)
        if risky:
            status = NCVNodeStatus.FAIL if any(result.status == "fail" for result in risky) else NCVNodeStatus.WARN
            result = risky[0]
            return self._node(
                NCVNode.SECURITY_RISK,
                status,
                "Security analyzer reported prompt injection, suspicious content, poisoning, retrieval anomaly, or privacy risk.",
                signals=[self._signal("security_result_status", result.status, result.analyzer_name, [], "Security analyzer status was consumed.")],
                recommended_fix=result.remediation or DEFAULT_REMEDIATIONS.get(result.failure_type or FailureType.SUSPICIOUS_CHUNK),
                method_type=NCVMethodType.EVIDENCE_AGGREGATION,
            )
        return self._node(
            NCVNode.SECURITY_RISK,
            NCVNodeStatus.PASS,
            "Security analyzer results reported no risk.",
            signals=[self._signal("security_results_available", len(security_results), "SecurityAnalyzers", [result.analyzer_name for result in security_results], "Security analyzer results were available.")],
            method_type=NCVMethodType.EVIDENCE_AGGREGATION,
        )

    def _build_result(
        self,
        run: RAGRun,
        node_results: list[NCVNodeResult],
        evidence_reports_used: list[str],
        missing_reports: list[str],
        fallback_heuristics_used: list[str],
    ) -> AnalyzerResult:
        first_failing_node = next((result.node for result in node_results if result.status == NCVNodeStatus.FAIL), None)
        first_uncertain_node = next((result.node for result in node_results if result.status == NCVNodeStatus.UNCERTAIN), None)
        passing_or_skipped = sum(1 for result in node_results if result.status in {NCVNodeStatus.PASS, NCVNodeStatus.SKIP})
        health_score = round(passing_or_skipped / len(node_results), 2) if node_results else None
        downstream_failure_chain = self._downstream_chain(node_results, first_failing_node)
        bottleneck = self._bottleneck_description(node_results, first_failing_node, first_uncertain_node)
        report = NCVReport(
            run_id=run.run_id,
            node_results=node_results,
            first_failing_node=first_failing_node,
            first_uncertain_node=first_uncertain_node,
            pipeline_health_score=health_score,
            bottleneck_description=bottleneck,
            downstream_failure_chain=downstream_failure_chain,
            evidence_reports_used=evidence_reports_used,
            missing_reports=missing_reports,
            fallback_heuristics_used=fallback_heuristics_used,
            method_type=self._report_method_type(node_results, first_failing_node),
            calibration_status=NCVCalibrationStatus.UNCALIBRATED,
            recommended_for_gating=False,
            limitations=[
                "Practical NCV-style architecture upgrade, not research-faithful NCV.",
                "Not calibrated for production gating.",
            ],
        )
        evidence = [report.model_dump_json(), report.bottleneck_description]

        if first_failing_node is not None:
            failure_type, stage = self._failure_mapping(first_failing_node, node_results)
            failing_result = next(result for result in node_results if result.node == first_failing_node)
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=failure_type,
                stage=stage,
                score=report.pipeline_health_score,
                evidence=evidence,
                ncv_report=report.model_dump(mode="json"),
                remediation=failing_result.recommended_fix or NODE_REMEDIATIONS.get(first_failing_node, DEFAULT_REMEDIATIONS[failure_type]),
            )

        if any(result.status in {NCVNodeStatus.WARN, NCVNodeStatus.UNCERTAIN} for result in node_results):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                score=report.pipeline_health_score,
                evidence=evidence,
                ncv_report=report.model_dump(mode="json"),
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            score=report.pipeline_health_score,
            evidence=evidence,
            ncv_report=report.model_dump(mode="json"),
        )

    def _node(
        self,
        node: NCVNode,
        status: NCVNodeStatus,
        reason: str,
        *,
        score: float | None = None,
        signals: list[NCVEvidenceSignal] | None = None,
        missing_evidence: list[str] | None = None,
        affected_claim_ids: list[str] | None = None,
        affected_chunk_ids: list[str] | None = None,
        affected_doc_ids: list[str] | None = None,
        alternative_explanations: list[str] | None = None,
        recommended_fix: str | None = None,
        method_type: NCVMethodType = NCVMethodType.PRACTICAL_APPROXIMATION,
        fallback_used: bool = False,
        limitations: list[str] | None = None,
    ) -> NCVNodeResult:
        return NCVNodeResult(
            node=node,
            status=status,
            node_score=score,
            primary_reason=reason,
            evidence_signals=signals or [],
            missing_evidence=missing_evidence or [],
            affected_claim_ids=affected_claim_ids or [],
            affected_chunk_ids=affected_chunk_ids or [],
            affected_doc_ids=affected_doc_ids or [],
            alternative_explanations=alternative_explanations or [],
            recommended_fix=recommended_fix,
            method_type=method_type,
            calibration_status=NCVCalibrationStatus.UNCALIBRATED,
            fallback_used=fallback_used,
            limitations=limitations or [],
        )

    def _skip(self, node: NCVNode, reason: str) -> NCVNodeResult:
        return self._node(node, NCVNodeStatus.SKIP, reason, method_type=NCVMethodType.PRACTICAL_APPROXIMATION)

    def _signal(
        self,
        name: str,
        value: str | float | int | bool | None,
        source_report: str | None,
        source_ids: list[str],
        interpretation: str,
        limitation: str | None = None,
    ) -> NCVEvidenceSignal:
        return NCVEvidenceSignal(
            signal_name=name,
            value=value,
            source_report=source_report,
            source_ids=source_ids,
            interpretation=interpretation,
            limitation=limitation,
        )

    def _result_index(
        self,
        results_or_index: AnalyzerResultIndex | list[AnalyzerResult],
    ) -> AnalyzerResultIndex:
        if isinstance(results_or_index, AnalyzerResultIndex):
            return results_or_index
        return AnalyzerResultIndex(results_or_index)

    def _get_result_by_name(
        self,
        results_or_index: AnalyzerResultIndex | list[AnalyzerResult],
        analyzer_name: str,
    ) -> AnalyzerResult | None:
        return self._result_index(results_or_index).by_name(analyzer_name)

    def _get_result_by_failure_type(
        self,
        results_or_index: AnalyzerResultIndex | list[AnalyzerResult],
        failure_type: FailureType,
    ) -> AnalyzerResult | None:
        return self._result_index(results_or_index).by_failure_type(failure_type)

    def _get_retrieval_diagnosis_report(self, run: RAGRun, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]):
        if run.retrieval_diagnosis_report is not None:
            return run.retrieval_diagnosis_report
        result = self._result_index(results_or_index).latest_with_field("retrieval_diagnosis_report")
        if result is not None:
            return result.retrieval_diagnosis_report
        return None

    def _get_retrieval_evidence_profile(self, run: RAGRun, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]):
        if run.retrieval_evidence_profile is not None:
            return run.retrieval_evidence_profile
        for result in self._result_index(results_or_index).all():
            bundle = result.grounding_evidence_bundle
            if bundle is not None and isinstance(bundle.metadata, dict) and bundle.metadata.get("retrieval_evidence_profile") is not None:
                return bundle.metadata["retrieval_evidence_profile"]
        return None

    def _get_sufficiency_result(self, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]):
        result = self._result_index(results_or_index).latest_with_field("sufficiency_result")
        if result is not None:
            return result.sufficiency_result
        return None

    def _get_claim_grounding_result(self, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]) -> AnalyzerResult | None:
        return self._result_index(results_or_index).grounding_result()

    def _get_citation_faithfulness_report(self, run: RAGRun, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]):
        if run.citation_faithfulness_report is not None:
            return run.citation_faithfulness_report
        result = self._result_index(results_or_index).latest_with_field("citation_faithfulness_report")
        if result is not None:
            return result.citation_faithfulness_report
        return None

    def _get_version_validity_report(self, run: RAGRun, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]):
        if run.version_validity_report is not None:
            return run.version_validity_report
        result = self._result_index(results_or_index).latest_with_field("version_validity_report")
        if result is not None:
            return result.version_validity_report
        return None

    def _get_parser_validation_results(self, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]) -> list[Any]:
        findings: list[Any] = []
        for result in self._result_index(results_or_index).all():
            if result.analyzer_name.lower().startswith("parser") or result.stage == FailureStage.PARSING:
                if result.diagnostic_rollup and isinstance(result.diagnostic_rollup.get("parser_findings"), list):
                    findings.extend(result.diagnostic_rollup["parser_findings"])
                elif result.diagnostic_rollup and isinstance(result.diagnostic_rollup.get("findings"), list):
                    findings.extend(result.diagnostic_rollup["findings"])
                elif result.status in {"fail", "warn"} and result.failure_type in {
                    FailureType.TABLE_STRUCTURE_LOSS,
                    FailureType.HIERARCHY_FLATTENING,
                    FailureType.METADATA_LOSS,
                    FailureType.PARSER_STRUCTURE_LOSS,
                    FailureType.CHUNKING_BOUNDARY_ERROR,
                }:
                    findings.append(result)
        return findings

    def _get_security_results(self, results_or_index: AnalyzerResultIndex | list[AnalyzerResult]) -> list[AnalyzerResult]:
        prior_results = self._result_index(results_or_index).all()
        security_failure_types = {
            FailureType.PROMPT_INJECTION,
            FailureType.SUSPICIOUS_CHUNK,
            FailureType.RETRIEVAL_ANOMALY,
            FailureType.PRIVACY_VIOLATION,
        }
        return [
            result
            for result in prior_results
            if result.stage == FailureStage.SECURITY
            or result.security_risk not in {None, SecurityRisk.NONE}
            or result.failure_type in security_failure_types
            or any(token in result.analyzer_name.lower() for token in ("security", "injection", "poison", "privacy", "anomaly"))
        ]

    def _get_claim_verifier_provider(self, grounding: AnalyzerResult) -> str | None:
        """Extract external claim verifier provider from grounding bundle or claim results."""
        bundle = getattr(grounding, "grounding_evidence_bundle", None)
        if bundle is not None:
            for rec in getattr(bundle, "claim_evidence_records", []):
                method = getattr(rec, "verifier_method", None)
                if method and method not in {"unknown", "heuristic"}:
                    return str(method)
        for claim in grounding.claim_results or []:
            method = getattr(claim, "verification_method", None)
            if method and method not in {"unknown", "heuristic", None}:
                return str(method)
        return None

    def _get_citation_verifier_provider(self, report: Any) -> str | None:
        """Extract external citation verifier provider from CitationFaithfulnessReport records."""
        for rec in getattr(report, "records", []):
            provider = getattr(rec, "external_signal_provider", None)
            if provider:
                return str(provider)
        return None

    def _unsupported_claims(self, result: AnalyzerResult) -> bool:
        return any(claim.label in {"unsupported", "contradicted", "abstain"} for claim in (result.claim_results or [])) or result.status == "fail"

    def _claim_ids(self, result: AnalyzerResult, *, unsupported_only: bool = False) -> list[str]:
        ids: list[str] = []
        for index, claim in enumerate(result.claim_results or [], start=1):
            if unsupported_only and claim.label not in {"unsupported", "contradicted"}:
                continue
            ids.append(f"claim-{index}")
        if result.sufficiency_result is not None:
            ids.extend(result.sufficiency_result.affected_claims)
        return self._unique(ids)

    def _affected_chunks_from_grounding(self, result: AnalyzerResult) -> list[str]:
        return self._affected_chunks_from_claim_results(result.claim_results or [])

    def _affected_chunks_from_claim_results(self, claims: Iterable[Any]) -> list[str]:
        chunk_ids: list[str] = []
        for claim in claims:
            chunk_ids.extend(getattr(claim, "supporting_chunk_ids", []) or [])
            chunk_ids.extend(getattr(claim, "candidate_chunk_ids", []) or [])
            chunk_ids.extend(getattr(claim, "contradicting_chunk_ids", []) or [])
        return self._unique(chunk_ids)

    def _parser_evidence_ids(self, finding: Any) -> list[str]:
        evidence = getattr(finding, "evidence", ()) or ()
        ids: list[str] = []
        for item in evidence:
            for attr in ("chunk_id", "element_id", "table_id"):
                value = getattr(item, attr, None)
                if value:
                    ids.append(value)
        return self._unique(ids)

    def _parser_finding_chunk_ids(self, findings: Iterable[Any]) -> list[str]:
        ids: list[str] = []
        for finding in findings:
            ids.extend(self._parser_evidence_ids(finding))
        return self._unique(ids)

    def _enum_value(self, value: Any) -> str:
        return getattr(value, "value", str(value))

    def _unique(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value and value not in seen:
                unique_values.append(value)
                seen.add(value)
        return unique_values

    def _downstream_chain(self, node_results: list[NCVNodeResult], first_failing_node: NCVNode | None) -> list[NCVNode]:
        if first_failing_node is None:
            return []
        after_failure = False
        chain: list[NCVNode] = []
        for result in node_results:
            if result.node == first_failing_node:
                after_failure = True
            if after_failure and result.status in {NCVNodeStatus.FAIL, NCVNodeStatus.WARN, NCVNodeStatus.UNCERTAIN}:
                chain.append(result.node)
        return chain

    def _bottleneck_description(
        self,
        node_results: list[NCVNodeResult],
        first_failing_node: NCVNode | None,
        first_uncertain_node: NCVNode | None,
    ) -> str:
        if first_failing_node is not None:
            node = next(result for result in node_results if result.node == first_failing_node)
            return f"Pipeline fails at {first_failing_node.value}: {node.primary_reason}"
        if first_uncertain_node is not None:
            node = next(result for result in node_results if result.node == first_uncertain_node)
            return f"Pipeline is uncertain at {first_uncertain_node.value}: {node.primary_reason}"
        warning = next((result for result in node_results if result.status == NCVNodeStatus.WARN), None)
        if warning is not None:
            return f"Pipeline warning at {warning.node.value}: {warning.primary_reason}"
        return "Pipeline is healthy across all evaluated NCV nodes"

    def _report_method_type(
        self,
        node_results: list[NCVNodeResult],
        first_failing_node: NCVNode | None,
    ) -> NCVMethodType:
        if first_failing_node is not None:
            failing = next(result for result in node_results if result.node == first_failing_node)
            if failing.method_type == NCVMethodType.EVIDENCE_AGGREGATION:
                return NCVMethodType.EVIDENCE_AGGREGATION
        has_evidence = any(result.method_type == NCVMethodType.EVIDENCE_AGGREGATION for result in node_results)
        has_fallback = any(result.fallback_used for result in node_results)
        if has_evidence and not has_fallback:
            return NCVMethodType.EVIDENCE_AGGREGATION
        if has_fallback and not has_evidence:
            return NCVMethodType.HEURISTIC_BASELINE
        return NCVMethodType.PRACTICAL_APPROXIMATION if has_evidence else NCVMethodType.HEURISTIC_BASELINE

    def _failure_mapping(self, node: NCVNode, node_results: list[NCVNodeResult]) -> tuple[FailureType, FailureStage]:
        if node == NCVNode.CLAIM_SUPPORT:
            result = next(item for item in node_results if item.node == node)
            if any(signal.value and "contradicted" in str(signal.value) for signal in result.evidence_signals):
                return FailureType.CONTRADICTED_CLAIM, FailureStage.GROUNDING
        if node == NCVNode.CITATION_SUPPORT:
            result = next(item for item in node_results if item.node == node)
            if any("phantom" in str(signal.value).lower() for signal in result.evidence_signals):
                return FailureType.CITATION_MISMATCH, FailureStage.RETRIEVAL
        return FAILURE_MAP[node]
