"""
Domain-agnostic retrieval-stage diagnosis analyzer.

RetrievalDiagnosisAnalyzerV0 aggregates existing evidence reports to classify
likely retrieval failure modes. It does not recompute retrieval quality, does
not require an LLM, and does not claim research-faithful RAGChecker, RAGAS,
DeepEval, RefChecker, Layer6, or A2P behavior.

v0 is a heuristic baseline, uncalibrated, and not recommended for production
gating.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from raggov.analyzers.base import BaseAnalyzer
from raggov.evaluators.base import ExternalEvaluationResult, ExternalSignalRecord
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType, SufficiencyResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.retrieval_diagnosis import (
    ClaimRetrievalDiagnosisRecord,
    RetrievalDiagnosisCalibrationStatus,
    RetrievalDiagnosisMethodType,
    RetrievalDiagnosisReport,
    RetrievalEvidenceSignal,
    RetrievalFailureType,
)
from raggov.models.retrieval_evidence import EvidenceRole, QueryRelevanceLabel, RelevanceMethod
from raggov.models.result_index import AnalyzerResultIndex
from raggov.models.run import RAGRun
from raggov.models.version_validity import DocumentValidityStatus, VersionValidityReport

_NO_CHUNKS_FIX = "Inspect retriever query, corpus coverage, top-k, filters, and index health."
_DEFAULT_FIX = "Review retrieval evidence, citation traceability, source validity, and ranking inputs."
_LIMITATIONS = [
    "v0 is a heuristic baseline and uncalibrated",
    "aggregates existing reports instead of recomputing retrieval quality",
    "not a research-faithful RAGChecker, RAGAS, DeepEval, RefChecker, Layer6, or A2P implementation",
    "recommended_for_gating is false until calibrated on labeled retrieval diagnosis data",
]

_INVALID_VERSION_STATUSES = {
    DocumentValidityStatus.SUPERSEDED,
    DocumentValidityStatus.WITHDRAWN,
    DocumentValidityStatus.EXPIRED,
    DocumentValidityStatus.NOT_YET_EFFECTIVE,
    DocumentValidityStatus.DEPRECATED,
    DocumentValidityStatus.REPLACED,
}


class RetrievalDiagnosisAnalyzerV0(BaseAnalyzer):
    """Aggregate upstream reports into a retrieval-stage failure diagnosis."""

    weight = 0.7

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        prior_results = self._prior_results()

        if not run.retrieved_chunks:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="no_retrieved_chunks",
                        value=True,
                        source_report="RAGRun",
                        source_ids=[],
                        interpretation="No chunks were returned by retrieval.",
                    )
                ],
                recommended_fix=_NO_CHUNKS_FIX,
            )
            return self._result("fail", report, FailureType.INSUFFICIENT_CONTEXT, _NO_CHUNKS_FIX)

        version_report = run.version_validity_report
        invalid_version_doc_ids = self._invalid_version_doc_ids(version_report)
        invalid_cited_doc_ids = sorted(set(invalid_version_doc_ids) & set(run.cited_doc_ids))
        if invalid_cited_doc_ids:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.VERSION_RETRIEVAL_FAILURE,
                invalid_cited_doc_ids=invalid_cited_doc_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="invalid_cited_source_docs",
                        value=len(invalid_cited_doc_ids),
                        source_report="VersionValidityReport",
                        source_ids=invalid_cited_doc_ids,
                        interpretation="The answer cited retrieved sources marked invalid by lifecycle metadata.",
                    )
                ],
                recommended_fix="Filter or demote invalid source versions and retrieve currently valid alternatives.",
            )
            return self._result("fail", report, FailureType.STALE_RETRIEVAL, report.recommended_fix)

        answer_bearing_invalid_doc_ids = self._answer_bearing_invalid_doc_ids(version_report)
        if answer_bearing_invalid_doc_ids:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.VERSION_RETRIEVAL_FAILURE,
                invalid_retrieved_doc_ids=answer_bearing_invalid_doc_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="answer_bearing_invalid_source_docs",
                        value=len(answer_bearing_invalid_doc_ids),
                        source_report="VersionValidityReport",
                        source_ids=answer_bearing_invalid_doc_ids,
                        interpretation="Answer-bearing retrieved sources are invalid by lifecycle metadata or freshness evidence.",
                    )
                ],
                recommended_fix="Filter invalid or stale answer-bearing sources before claim generation and support checking.",
            )
            return self._result("fail", report, FailureType.STALE_RETRIEVAL, report.recommended_fix)

        retrieval_quality_affected_doc_ids = self._retrieval_quality_affected_doc_ids(version_report)
        if retrieval_quality_affected_doc_ids:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.VERSION_RETRIEVAL_FAILURE,
                invalid_retrieved_doc_ids=retrieval_quality_affected_doc_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="retrieval_quality_affected_by_stale_source_docs",
                        value=len(retrieval_quality_affected_doc_ids),
                        source_report="VersionValidityReport",
                        source_ids=retrieval_quality_affected_doc_ids,
                        interpretation="Retrieved stale sources are relevant enough to pollute ranking, deduplication, or provenance even though they were not cited.",
                    )
                ],
                recommended_fix="Deduplicate or demote stale retrieved sources so they do not crowd out current evidence.",
            )
            return self._result("fail", report, FailureType.STALE_RETRIEVAL, report.recommended_fix)

        retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
        invalid_retrieved_doc_ids = sorted(set(invalid_version_doc_ids) & retrieved_doc_ids)
        if invalid_retrieved_doc_ids:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.VERSION_RETRIEVAL_FAILURE,
                invalid_retrieved_doc_ids=invalid_retrieved_doc_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="invalid_retrieved_source_docs",
                        value=len(invalid_retrieved_doc_ids),
                        source_report="VersionValidityReport",
                        source_ids=invalid_retrieved_doc_ids,
                        interpretation="Retrieval surfaced sources marked invalid by lifecycle metadata.",
                    )
                ],
                alternative_explanations=[
                    "retrieval surfaced invalid sources, but answer did not cite them"
                ],
                recommended_fix="Filter invalid source versions at retrieval time or rank valid alternatives above them.",
            )
            return self._result("warn", report, FailureType.STALE_RETRIEVAL, report.recommended_fix)

        phantom_citation_doc_ids = self._phantom_citation_doc_ids(run, prior_results)
        if phantom_citation_doc_ids:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.CITATION_RETRIEVAL_MISMATCH,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="phantom_citation_doc_ids",
                        value=len(phantom_citation_doc_ids),
                        source_report=self._phantom_source_report(run),
                        source_ids=phantom_citation_doc_ids,
                        interpretation="Cited documents were absent from retrieved evidence.",
                    )
                ],
                recommended_fix="Audit citation selection and ensure cited documents are present in retrieved context.",
            )
            return self._result("fail", report, FailureType.CITATION_MISMATCH, report.recommended_fix)

        missing_citation_claim_ids = self._missing_citation_claim_ids(run)
        sufficiency = self._sufficiency_result(prior_results)
        if missing_citation_claim_ids and not self._sufficiency_is_insufficient(sufficiency):
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.CITATION_RETRIEVAL_MISMATCH,
                affected_claim_ids=missing_citation_claim_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="missing_citation_claim_ids",
                        value=len(missing_citation_claim_ids),
                        source_report="CitationFaithfulnessReport",
                        source_ids=missing_citation_claim_ids,
                        interpretation="One or more claims lack source citations.",
                    )
                ],
                recommended_fix="Require claim-level citation links to retrieved evidence before answer finalization.",
            )
            return self._result("warn", report, FailureType.CITATION_MISMATCH, report.recommended_fix)

        unsupported_claims = self._unsupported_claim_records(prior_results)
        if unsupported_claims and self._externally_relevant_rank_failure(run, unsupported_claims):
            candidate_chunk_ids = self._candidate_chunk_ids(unsupported_claims)
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RANK_FAILURE_UNKNOWN,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="external_relevant_candidates_with_unsupported_claims",
                        value=len(candidate_chunk_ids),
                        source_report="GroundingEvidenceBundle+RetrievalEvidenceProfile",
                        source_ids=candidate_chunk_ids,
                        interpretation=(
                            "External relevance labels marked candidate chunks relevant, "
                            "but claim grounding still found unsupported claims."
                        ),
                        limitation="External relevance labels are uncalibrated and advisory.",
                    )
                ],
                recommended_fix="Inspect rank positions, chunk boundaries, and context assembly for relevant-but-insufficient evidence.",
            )
            return self._result("warn", report, FailureType.RERANKER_FAILURE, report.recommended_fix)

        if unsupported_claims and self._query_attribute_missing_but_entity_present(run):
            affected_claim_ids = [claim.claim_id for claim in unsupported_claims]
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
                affected_claim_ids=affected_claim_ids,
                candidate_chunk_ids=self._candidate_chunk_ids(unsupported_claims),
                claim_records=[
                    self._claim_record(
                        claim,
                        RetrievalFailureType.RETRIEVAL_MISS,
                        "Retrieved chunks mention the requested entity but omit the requested attribute.",
                    )
                    for claim in unsupported_claims
                ],
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="entity_present_attribute_missing",
                        value=True,
                        source_report="RAGRun+GroundingEvidenceBundle",
                        source_ids=affected_claim_ids,
                        interpretation=(
                            "Retrieved chunks are near misses: they mention the entity but not the attribute needed by the query."
                        ),
                    )
                ],
                recommended_fix="Retrieve evidence for the missing entity attribute before answering.",
            )
            return self._result("fail", report, FailureType.INSUFFICIENT_CONTEXT, report.recommended_fix)

        if (
            sufficiency is not None
            and unsupported_claims
            and (not run.cited_doc_ids or self._externally_irrelevant_retrieval_miss(run))
            and (
                sufficiency.sufficiency_label == "insufficient"
                or not sufficiency.sufficient
            )
        ) or (unsupported_claims and self._externally_irrelevant_retrieval_miss(run)):
            claim_records = [
                self._claim_record(
                    claim,
                    RetrievalFailureType.RETRIEVAL_MISS,
                    "Required evidence appears missing from retrieved context.",
                )
                for claim in unsupported_claims
            ]
            affected_claim_ids = self._unique(
                [claim.claim_id for claim in unsupported_claims] + (sufficiency.affected_claims if sufficiency else [])
            )
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
                affected_claim_ids=affected_claim_ids,
                candidate_chunk_ids=self._candidate_chunk_ids(unsupported_claims),
                claim_records=claim_records,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="insufficient_context_with_unsupported_claims",
                        value=True,
                        source_report="SufficiencyResult+GroundingEvidenceBundle",
                        source_ids=affected_claim_ids,
                        interpretation="Sufficiency and grounding both indicate missing support.",
                    )
                ],
                recommended_fix="Expand retrieval for missing evidence requirements and unsupported claims.",
            )
            return self._result("fail", report, FailureType.INSUFFICIENT_CONTEXT, report.recommended_fix)

        if (
            sufficiency is not None
            and not unsupported_claims
            and (
                sufficiency.sufficiency_label == "insufficient"
                or not sufficiency.sufficient
            )
        ):
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RETRIEVAL_MISS,
                affected_claim_ids=list(sufficiency.affected_claims),
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="insufficient_context_without_verifiable_claims",
                        value=True,
                        source_report="SufficiencyResult",
                        source_ids=list(sufficiency.affected_claims),
                        interpretation="Retrieved context does not cover critical query requirements before claim grounding can verify an answer.",
                    )
                ],
                recommended_fix="Expand retrieval for missing query requirements before generating an answer.",
            )
            return self._result("fail", report, FailureType.INSUFFICIENT_CONTEXT, report.recommended_fix)

        profile = run.retrieval_evidence_profile
        if profile is not None and profile.noisy_chunk_ids:
            noisy_chunk_ids = list(profile.noisy_chunk_ids)
            noise_ratio = len(noisy_chunk_ids) / max(len(run.retrieved_chunks), 1)
            noise_ratio_warn = float(self.config.get("noise_ratio_warn", 0.5))
            noise_min_chunks = int(self.config.get("noise_min_chunks", 2))
            noise_ratio_fail = float(self.config.get("noise_ratio_fail", 0.9))
            noise_fail_min_chunks = int(self.config.get("noise_fail_min_chunks", 5))
            status = (
                "fail"
                if noise_ratio >= noise_ratio_fail
                and len(noisy_chunk_ids) >= noise_fail_min_chunks
                else "warn"
            )
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RETRIEVAL_NOISE,
                noisy_chunk_ids=noisy_chunk_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="noisy_chunk_ids",
                        value=len(noisy_chunk_ids),
                        source_report="RetrievalEvidenceProfile",
                        source_ids=noisy_chunk_ids,
                        interpretation="Existing retrieval evidence profile marked chunks as noise.",
                        limitation="Thresholds are heuristic and uncalibrated.",
                    )
                ],
                recommended_fix="Tune retriever filters, query rewriting, and ranking to reduce noisy chunks.",
                limitations=[
                    f"noise_ratio_warn={noise_ratio_warn}",
                    f"noise_min_chunks={noise_min_chunks}",
                    f"noise_ratio_fail={noise_ratio_fail}",
                    f"noise_fail_min_chunks={noise_fail_min_chunks}",
                ],
            )
            return self._result(status, report, FailureType.RETRIEVAL_ANOMALY, report.recommended_fix)

        if self._rank_failure_unknown(run, unsupported_claims):
            candidate_chunk_ids = self._candidate_chunk_ids(unsupported_claims)
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.RANK_FAILURE_UNKNOWN,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_signals=[
                    RetrievalEvidenceSignal(
                        signal_name="candidate_evidence_without_rank_labels",
                        value=len(candidate_chunk_ids),
                        source_report="GroundingEvidenceBundle+RetrievalEvidenceProfile",
                        source_ids=candidate_chunk_ids,
                        interpretation=(
                            "Evidence may exist, but ranking failure cannot be confirmed "
                            "without rank or relevance labels."
                        ),
                    )
                ],
                recommended_fix="Capture rank positions, relevance labels, and reranker decisions for candidate evidence.",
            )
            return self._result("warn", report, FailureType.RERANKER_FAILURE, report.recommended_fix)

        missing_reports = self._missing_reports(run, prior_results)
        if missing_reports:
            report = self._report(
                run=run,
                primary_failure_type=RetrievalFailureType.INSUFFICIENT_EVIDENCE_TO_DIAGNOSE,
                missing_reports=missing_reports,
                evidence_signals=[],
                recommended_fix="Run retrieval evidence, grounding, sufficiency, citation, and source-validity analyzers.",
            )
            return self._result("warn", report, None, report.recommended_fix)

        report = self._report(
            run=run,
            primary_failure_type=RetrievalFailureType.NO_CLEAR_RETRIEVAL_FAILURE,
            recommended_fix="No retrieval-stage fix indicated by available v0 evidence.",
        )
        return self._result("pass", report, None, report.recommended_fix)

    def _report(
        self,
        *,
        run: RAGRun,
        primary_failure_type: RetrievalFailureType,
        recommended_fix: str,
        affected_claim_ids: list[str] | None = None,
        supporting_chunk_ids: list[str] | None = None,
        candidate_chunk_ids: list[str] | None = None,
        noisy_chunk_ids: list[str] | None = None,
        invalid_retrieved_doc_ids: list[str] | None = None,
        invalid_cited_doc_ids: list[str] | None = None,
        missing_reports: list[str] | None = None,
        claim_records: list[ClaimRetrievalDiagnosisRecord] | None = None,
        evidence_signals: list[RetrievalEvidenceSignal] | None = None,
        alternative_explanations: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> RetrievalDiagnosisReport:
        return RetrievalDiagnosisReport(
            run_id=run.run_id,
            primary_failure_type=primary_failure_type,
            affected_claim_ids=affected_claim_ids or [],
            supporting_chunk_ids=supporting_chunk_ids or [],
            candidate_chunk_ids=candidate_chunk_ids or [],
            noisy_chunk_ids=noisy_chunk_ids or [],
            invalid_retrieved_doc_ids=invalid_retrieved_doc_ids or [],
            invalid_cited_doc_ids=invalid_cited_doc_ids or [],
            missing_reports=missing_reports or [],
            claim_records=claim_records or [],
            evidence_signals=(evidence_signals or []) + self._external_retrieval_evidence_signals(run),
            alternative_explanations=alternative_explanations or [],
            recommended_fix=recommended_fix,
            method_type=RetrievalDiagnosisMethodType.HEURISTIC_BASELINE,
            calibration_status=RetrievalDiagnosisCalibrationStatus.UNCALIBRATED,
            recommended_for_gating=False,
            limitations=list(_LIMITATIONS) + (limitations or []),
        )

    def _result(
        self,
        status: str,
        report: RetrievalDiagnosisReport,
        failure_type: FailureType | None,
        remediation: str,
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name(),
            status=status,  # type: ignore[arg-type]
            failure_type=failure_type,
            stage=FailureStage.RETRIEVAL if status in {"fail", "warn"} else None,
            evidence=[report.model_dump_json()],
            remediation=remediation if status in {"fail", "warn"} else None,
            retrieval_diagnosis_report=report,
        )

    def _prior_results(self) -> list[AnalyzerResult]:
        raw = self.config.get("prior_results", [])
        return [item for item in raw if isinstance(item, AnalyzerResult)]

    def _invalid_version_doc_ids(self, report: VersionValidityReport | None) -> list[str]:
        if report is None:
            return []
        ids = (
            list(report.superseded_doc_ids)
            + list(report.withdrawn_doc_ids)
            + list(report.expired_doc_ids)
            + list(report.not_yet_effective_doc_ids)
            + list(getattr(report, "deprecated_doc_ids", []))
            + list(getattr(report, "replaced_doc_ids", []))
        )
        for record in report.document_records:
            if record.validity_status in _INVALID_VERSION_STATUSES:
                ids.append(record.doc_id)
        return sorted(set(ids))

    def _answer_bearing_invalid_doc_ids(self, report: VersionValidityReport | None) -> list[str]:
        if report is None:
            return []
        return list(getattr(report, "answer_bearing_invalid_doc_ids", []))

    def _retrieval_quality_affected_doc_ids(self, report: VersionValidityReport | None) -> list[str]:
        if report is None:
            return []
        return list(getattr(report, "retrieval_quality_affected_doc_ids", []))

    def _phantom_citation_doc_ids(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
    ) -> list[str]:
        ids: list[str] = []
        result_index = AnalyzerResultIndex(prior_results)
        if run.citation_faithfulness_report is not None:
            ids.extend(run.citation_faithfulness_report.phantom_citation_doc_ids)
        if run.retrieval_evidence_profile is not None:
            ids.extend(run.retrieval_evidence_profile.phantom_citation_doc_ids)
        mismatch_result = result_index.by_name("CitationMismatchAnalyzer")
        if mismatch_result is not None and not ids:
            retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
            ids.extend(doc_id for doc_id in run.cited_doc_ids if doc_id not in retrieved_doc_ids)
        return sorted(set(ids))

    def _phantom_source_report(self, run: RAGRun) -> str:
        if run.citation_faithfulness_report is not None:
            return "CitationFaithfulnessReport"
        if run.retrieval_evidence_profile is not None:
            return "RetrievalEvidenceProfile"
        return "CitationMismatchAnalyzer"

    def _missing_citation_claim_ids(self, run: RAGRun) -> list[str]:
        if run.citation_faithfulness_report is None:
            return []
        return list(run.citation_faithfulness_report.missing_citation_claim_ids)

    def _sufficiency_result(self, prior_results: list[AnalyzerResult]) -> SufficiencyResult | None:
        fallback: SufficiencyResult | None = None
        for result in reversed(prior_results):
            if result.sufficiency_result is not None:
                sufficiency = result.sufficiency_result
                if fallback is None:
                    fallback = sufficiency
                if self._sufficiency_is_insufficient(sufficiency):
                    return sufficiency
        return fallback

    def _sufficiency_is_insufficient(self, sufficiency: SufficiencyResult | None) -> bool:
        return bool(
            sufficiency is not None
            and (
                sufficiency.sufficiency_label == "insufficient"
                or (not sufficiency.sufficient and bool(sufficiency.coverage))
            )
        )

    def _sufficiency_has_critical_anchor(self, sufficiency: SufficiencyResult | None) -> bool:
        if sufficiency is None:
            return False
        for coverage in sufficiency.coverage:
            rationale = coverage.rationale.lower()
            if "project " in rationale or "tool " in rationale or "product " in rationale:
                return True
        return False

    def _unsupported_claim_records(
        self,
        prior_results: list[AnalyzerResult],
    ) -> list[ClaimEvidenceRecord]:
        records: list[ClaimEvidenceRecord] = []
        for result in prior_results:
            bundle = result.grounding_evidence_bundle
            if bundle is not None:
                records.extend(
                    record
                    for record in bundle.claim_evidence_records
                    if record.verification_label
                    in {ClaimVerificationLabel.INSUFFICIENT, ClaimVerificationLabel.UNVERIFIED}
                )
            for claim in result.claim_results or []:
                if claim.label == "unsupported":
                    records.append(
                        ClaimEvidenceRecord(
                            claim_id=claim.claim_text,
                            claim_text=claim.claim_text,
                            candidate_evidence_chunk_ids=list(claim.candidate_chunk_ids),
                            verification_label=ClaimVerificationLabel.INSUFFICIENT,
                        )
                    )
        return records

    def _claim_record(
        self,
        claim: ClaimEvidenceRecord,
        failure_type: RetrievalFailureType,
        explanation: str,
    ) -> ClaimRetrievalDiagnosisRecord:
        signal = RetrievalEvidenceSignal(
            signal_name="unsupported_claim",
            value=claim.verification_label.value,
            source_report="GroundingEvidenceBundle",
            source_ids=[claim.claim_id],
            interpretation=explanation,
        )
        return ClaimRetrievalDiagnosisRecord(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            retrieval_failure_type=failure_type,
            supporting_chunk_ids=list(claim.cited_chunk_ids),
            candidate_chunk_ids=list(claim.candidate_evidence_chunk_ids),
            evidence_signals=[signal],
            explanation=explanation,
        )

    def _candidate_chunk_ids(self, claims: Iterable[ClaimEvidenceRecord]) -> list[str]:
        ids: list[str] = []
        for claim in claims:
            ids.extend(claim.candidate_evidence_chunk_ids)
        return self._unique(ids)

    def _rank_failure_unknown(
        self,
        run: RAGRun,
        unsupported_claims: list[ClaimEvidenceRecord],
    ) -> bool:
        profile = run.retrieval_evidence_profile
        candidate_chunk_ids = set(self._candidate_chunk_ids(unsupported_claims))
        if profile is None or not candidate_chunk_ids:
            return False
        relevant_chunks = [
            chunk
            for chunk in profile.chunks
            if chunk.chunk_id in candidate_chunk_ids
            or chunk.evidence_role
            in {EvidenceRole.NECESSARY_SUPPORT, EvidenceRole.PARTIAL_SUPPORT}
        ]
        if not relevant_chunks:
            return False
        # Original path: relevance unknown — cannot confirm rank failure.
        if all(
            chunk.query_relevance_label == QueryRelevanceLabel.UNKNOWN
            and chunk.query_relevance_score is None
            for chunk in relevant_chunks
        ):
            return True
        # External signal path: cross-encoder labeled chunks as relevant but
        # claims are still unsupported — ranking failed to surface the right
        # content for each claim, even though content was retrieved.
        return all(
            chunk.query_relevance_label
            in {QueryRelevanceLabel.RELEVANT, QueryRelevanceLabel.PARTIAL}
            and chunk.relevance_method == RelevanceMethod.CROSS_ENCODER
            for chunk in relevant_chunks
        )

    def _externally_relevant_rank_failure(
        self,
        run: RAGRun,
        unsupported_claims: list[ClaimEvidenceRecord],
    ) -> bool:
        profile = run.retrieval_evidence_profile
        candidate_chunk_ids = set(self._candidate_chunk_ids(unsupported_claims))
        if profile is None or not candidate_chunk_ids:
            return False
        relevant_chunks = [
            chunk
            for chunk in profile.chunks
            if chunk.chunk_id in candidate_chunk_ids
        ]
        if not relevant_chunks:
            return False
        # External signal path: cross-encoder labeled chunks as relevant/partial but
        # claims are still unsupported — ranking failed to surface the right
        # content for each claim, even though content was retrieved.
        return any(
            chunk.query_relevance_label
            in {QueryRelevanceLabel.RELEVANT, QueryRelevanceLabel.PARTIAL}
            and chunk.relevance_method == RelevanceMethod.CROSS_ENCODER
            for chunk in relevant_chunks
        )

    def _externally_irrelevant_retrieval_miss(self, run: RAGRun) -> bool:
        """Check if external signals explicitly mark all retrieved chunks as irrelevant."""
        profile = run.retrieval_evidence_profile
        if profile is None or not profile.chunks:
            return False
            
        has_external = any(
            cp.relevance_method == RelevanceMethod.CROSS_ENCODER
            for cp in profile.chunks
        )
        if not has_external:
            return False
            
        # If external tool exists and all chunks it scored are IRRELEVANT, it's a retrieval miss.
        scored_chunks = [cp for cp in profile.chunks if cp.relevance_method == RelevanceMethod.CROSS_ENCODER]
        return all(cp.query_relevance_label == QueryRelevanceLabel.IRRELEVANT for cp in scored_chunks)

    def _query_attribute_missing_but_entity_present(self, run: RAGRun) -> bool:
        query_terms = self._terms(run.query)
        context_terms = self._terms(" ".join(chunk.text for chunk in run.retrieved_chunks))
        if not query_terms or not context_terms:
            return False
        attribute_terms = {"salary", "income", "rate", "interest", "deadline", "date", "amount"}
        entity_terms = {"ceo", "cfo", "manager", "project", "account", "company"}
        requested_attributes = query_terms & attribute_terms
        if not requested_attributes:
            return False
        return bool(query_terms & entity_terms & context_terms) and not bool(requested_attributes & context_terms)

    def _missing_reports(
        self,
        run: RAGRun,
        prior_results: list[AnalyzerResult],
    ) -> list[str]:
        no_claims_required = self._claim_grounding_not_required(prior_results)
        missing: list[str] = []
        if run.retrieval_evidence_profile is None:
            missing.append("retrieval_evidence_profile")
        if run.citation_faithfulness_report is None and not no_claims_required:
            missing.append("citation_faithfulness_report")
        if run.version_validity_report is None:
            missing.append("version_validity_report")
        if self._sufficiency_result(prior_results) is None:
            missing.append("sufficiency_result")
        if (
            not no_claims_required
            and not any(result.grounding_evidence_bundle is not None for result in prior_results)
        ):
            missing.append("grounding_evidence_bundle")
            
        # Check for missing external retrieval signals in external-enhanced mode
        if self._external_retrieval_signal_required():
            profile = run.retrieval_evidence_profile
            has_external = profile is not None and any(
                cp.relevance_method == RelevanceMethod.CROSS_ENCODER for cp in profile.chunks
            )
            if not has_external:
                missing.append("external_retrieval_relevance_signal")
                
        return missing

    def _claim_grounding_not_required(self, prior_results: list[AnalyzerResult]) -> bool:
        grounding = AnalyzerResultIndex(prior_results).by_name("ClaimGroundingAnalyzer")
        if grounding is None:
            return False
        return grounding.status == "skip" and any(
            "no claims extracted" in evidence for evidence in grounding.evidence
        )

    def _external_retrieval_signal_required(self) -> bool:
        if self.config.get("mode") != "external-enhanced":
            return False
        if self.config.get("retrieval_relevance_provider") == "cross_encoder":
            return True
        enabled = set(self.config.get("enabled_external_providers", []))
        return "cross_encoder_relevance" in enabled

    def _unique(self, values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _terms(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _external_retrieval_evidence_signals(self, run: RAGRun) -> list[RetrievalEvidenceSignal]:
        records: list[ExternalSignalRecord] = []
        raw_results = run.metadata.get("external_evaluation_results", [])
        for item in raw_results:
            if isinstance(item, ExternalEvaluationResult):
                records.extend(item.signals)
            elif isinstance(item, ExternalSignalRecord):
                records.append(item)
            elif isinstance(item, dict):
                if "signals" in item:
                    try:
                        result = ExternalEvaluationResult.model_validate(item)
                    except Exception:
                        continue
                    records.extend(result.signals)
                else:
                    try:
                        records.append(ExternalSignalRecord.model_validate(item))
                    except Exception:
                        continue

        profile = run.retrieval_evidence_profile
        if profile is not None:
            for item in getattr(profile, "external_signals", []):
                try:
                    records.append(ExternalSignalRecord.model_validate(item))
                except Exception:
                    continue

        return [
            RetrievalEvidenceSignal(
                signal_name=record.metric_name,
                value=record.value,
                source_report=f"ExternalEvaluationResult:{record.provider.value}",
                source_ids=list(record.affected_chunk_ids or record.evidence_ids),
                interpretation=self._external_signal_interpretation(record),
                limitation=(
                    "External retrieval/context signal is uncalibrated locally, "
                    "recommended_for_gating=false, and cannot determine GovRAG diagnosis alone."
                ),
            )
            for record in records
            if record.provider.value in {"ragas", "deepeval", "cross_encoder", "ragchecker"}
        ]

    def _external_signal_interpretation(self, record: ExternalSignalRecord) -> str:
        metric = record.metric_name
        value = record.value
        if "recall" in metric:
            return f"{metric}={value}; low recall is advisory evidence for retrieval_miss."
        if "precision" in metric and "contextual" in metric:
            return f"{metric}={value}; poor contextual precision is advisory evidence for rank_failure_unknown."
        if "precision" in metric or "relevancy" in metric:
            return f"{metric}={value}; low precision/relevancy is advisory evidence for retrieval_noise."
        if "faithfulness" in metric:
            return (
                f"{metric}={value}; faithfulness is advisory grounding evidence and "
                "does not override GovRAG claim/citation verifiers."
            )
        if "utilization" in metric:
            return f"{metric}={value}; low context utilization suggests generation may have ignored retrieved evidence."
        if "hallucination" in metric:
            return f"{metric}={value}; high hallucination is advisory evidence for generation/claim issues, not retrieval alone."
        return f"{metric}={value}; external adapter evidence only."
