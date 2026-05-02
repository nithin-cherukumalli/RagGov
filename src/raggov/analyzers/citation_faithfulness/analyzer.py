"""
CitationFaithfulnessAnalyzerV0.

This analyzer builds a practical-approximation report from existing claim
grounding records and retrieval evidence. It does not run an LLM judge,
counterfactual probe, RefChecker, RAGChecker, or Wallat-style citation
faithfulness method.
"""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.citation_faithfulness import (
    CitationCalibrationStatus,
    CitationEvidenceSource,
    CitationFaithfulnessReport,
    CitationFaithfulnessRisk,
    CitationMethodType,
    CitationSupportLabel,
    ClaimCitationFaithfulnessRecord,
)
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.grounding import (
    ClaimEvidenceRecord,
    ClaimVerificationLabel,
    GroundingEvidenceBundle,
)
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
from raggov.models.run import RAGRun


_SKIP_REASON = "no claim evidence available for citation faithfulness analysis"
_REMEDIATION = "Review claim citations and ensure cited sources support each claim."
_LIMITATIONS = [
    "v0 checks citation support using existing claim grounding and retrieval evidence only",
    "v0 does not prove the model genuinely relied on the cited source",
    "v0 does not perform counterfactual citation faithfulness probes",
    "v0 is not a research-faithful RefChecker or RAGChecker implementation",
]


class CitationFaithfulnessAnalyzerV0(BaseAnalyzer):
    """
    Evaluate whether cited sources support claim-level evidence records.

    Inputs are precomputed claim grounding records plus optional retrieval
    evidence on the RAGRun. This analyzer intentionally does not build claim
    evidence itself, so it does not modify or invoke claim grounding internals.
    """

    weight = 0.7

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        records = self._claim_evidence_records(run)
        if not run.final_answer or not records:
            return self.skip(_SKIP_REASON)

        report = self._build_report(run, records)
        evidence = self._evidence(report)

        if report.phantom_citation_doc_ids or report.contradicted_claim_ids:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=(
                    FailureType.CONTRADICTED_CLAIM
                    if report.contradicted_claim_ids
                    else FailureType.CITATION_MISMATCH
                ),
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=_REMEDIATION,
                citation_faithfulness_report=report,
            )

        if report.unsupported_claim_ids or report.missing_citation_claim_ids:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.CITATION_MISMATCH,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                remediation=_REMEDIATION,
                citation_faithfulness_report=report,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            citation_faithfulness_report=report,
        )

    def _build_report(
        self,
        run: RAGRun,
        records: list[ClaimEvidenceRecord],
    ) -> CitationFaithfulnessReport:
        built_records = [self._build_record(run, record) for record in records]

        return CitationFaithfulnessReport(
            run_id=run.run_id,
            records=built_records,
            unsupported_claim_ids=[
                r.claim_id
                for r in built_records
                if r.citation_support_label == CitationSupportLabel.UNSUPPORTED
            ],
            phantom_citation_doc_ids=list(
                dict.fromkeys(
                    doc_id
                    for r in built_records
                    if r.citation_support_label == CitationSupportLabel.CITATION_PHANTOM
                    for doc_id in r.cited_doc_ids
                    if doc_id not in self._retrieved_doc_ids(run)
                )
            ),
            missing_citation_claim_ids=[
                r.claim_id
                for r in built_records
                if r.citation_support_label == CitationSupportLabel.CITATION_MISSING
            ],
            contradicted_claim_ids=[
                r.claim_id
                for r in built_records
                if r.citation_support_label == CitationSupportLabel.CONTRADICTED
            ],
            claim_grounding_used=True,
            retrieval_evidence_profile_used=run.retrieval_evidence_profile is not None,
            legacy_citation_fallback_used=any(
                r.evidence_source == CitationEvidenceSource.LEGACY_CITATION_IDS
                for r in built_records
            ),
            method_type=CitationMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=CitationCalibrationStatus.UNCALIBRATED,
            recommended_for_gating=False,
            limitations=list(_LIMITATIONS),
        )

    def _build_record(
        self,
        run: RAGRun,
        claim: ClaimEvidenceRecord,
    ) -> ClaimCitationFaithfulnessRecord:
        cited_doc_ids, cited_chunk_ids, evidence_source = self._citation_sources(
            run, claim
        )
        supporting_chunk_ids = self._ids_from_record(
            claim, "supporting_chunk_ids", "candidate_evidence_chunk_ids"
        )
        contradicted_chunk_ids = self._ids_from_record(
            claim, "contradicting_chunk_ids", "contradicted_by_chunk_ids"
        )
        neutral_chunk_ids = self._ids_from_record(claim, "neutral_chunk_ids")
        profile_support, profile_contradictions, profile_neutral = (
            self._profile_chunk_ids_for_claim(run.retrieval_evidence_profile, claim.claim_id)
        )
        supporting_chunk_ids = list(dict.fromkeys(supporting_chunk_ids + profile_support))
        contradicted_chunk_ids = list(
            dict.fromkeys(contradicted_chunk_ids + profile_contradictions)
        )
        neutral_chunk_ids = list(dict.fromkeys(neutral_chunk_ids + profile_neutral))

        cited_chunk_ids = self._expand_cited_chunks(run, cited_doc_ids, cited_chunk_ids)
        label, explanation = self._support_label(
            run=run,
            claim=claim,
            cited_doc_ids=cited_doc_ids,
            cited_chunk_ids=cited_chunk_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            contradicted_chunk_ids=contradicted_chunk_ids,
        )

        return ClaimCitationFaithfulnessRecord(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            cited_doc_ids=cited_doc_ids,
            cited_chunk_ids=cited_chunk_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            contradicted_by_chunk_ids=contradicted_chunk_ids,
            neutral_chunk_ids=neutral_chunk_ids,
            citation_support_label=label,
            faithfulness_risk=self._risk(label),
            evidence_source=evidence_source,
            explanation=explanation,
            limitations=list(_LIMITATIONS),
        )

    def _support_label(
        self,
        *,
        run: RAGRun,
        claim: ClaimEvidenceRecord,
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
        supporting_chunk_ids: list[str],
        contradicted_chunk_ids: list[str],
    ) -> tuple[CitationSupportLabel, str | None]:
        if not cited_doc_ids and not cited_chunk_ids:
            return CitationSupportLabel.CITATION_MISSING, "no citation available for claim"

        retrieved_doc_ids = self._retrieved_doc_ids(run)
        phantom_doc_ids = [doc_id for doc_id in cited_doc_ids if doc_id not in retrieved_doc_ids]
        if phantom_doc_ids:
            return (
                CitationSupportLabel.CITATION_PHANTOM,
                f"cited document not present in retrieved context: {', '.join(phantom_doc_ids)}",
            )

        if self._has_cited_contradiction(
            run, claim.claim_id, cited_doc_ids, cited_chunk_ids, contradicted_chunk_ids
        ):
            return CitationSupportLabel.CONTRADICTED, "cited source contradicts the claim"

        support_overlap = set(supporting_chunk_ids) & set(cited_chunk_ids)
        if supporting_chunk_ids and support_overlap:
            if support_overlap == set(supporting_chunk_ids):
                return CitationSupportLabel.FULLY_SUPPORTED, "cited source overlaps all supporting chunks"
            return CitationSupportLabel.PARTIALLY_SUPPORTED, "cited source overlaps some supporting chunks"

        if self._is_supported_claim(claim) and supporting_chunk_ids:
            return (
                CitationSupportLabel.UNSUPPORTED,
                "claim appears supported by retrieved context, but not by cited source",
            )

        if self._is_unsupported_claim(claim):
            return CitationSupportLabel.UNSUPPORTED, "claim is not supported by existing claim grounding evidence"

        return CitationSupportLabel.UNKNOWN, "citation support unavailable from existing evidence"

    def _citation_sources(
        self, run: RAGRun, claim: ClaimEvidenceRecord
    ) -> tuple[list[str], list[str], CitationEvidenceSource]:
        claim_cited_docs = self._ids_from_record(claim, "cited_doc_ids")
        claim_cited_chunks = self._ids_from_record(claim, "cited_chunk_ids")
        if claim_cited_docs or claim_cited_chunks:
            return claim_cited_docs, claim_cited_chunks, CitationEvidenceSource.CLAIM_GROUNDING

        run_cited_doc_ids = list(run.cited_doc_ids)
        run_cited_chunk_ids = [str(v) for v in run.metadata.get("cited_chunk_ids", [])]
        if run_cited_doc_ids or run_cited_chunk_ids:
            return (
                run_cited_doc_ids,
                run_cited_chunk_ids,
                CitationEvidenceSource.LEGACY_CITATION_IDS,
            )

        return [], [], CitationEvidenceSource.UNAVAILABLE

    def _expand_cited_chunks(
        self,
        run: RAGRun,
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
    ) -> list[str]:
        expanded = list(cited_chunk_ids)
        for chunk in run.retrieved_chunks:
            if chunk.source_doc_id in cited_doc_ids:
                expanded.append(chunk.chunk_id)
        return list(dict.fromkeys(expanded))

    def _has_cited_contradiction(
        self,
        run: RAGRun,
        claim_id: str,
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
        contradicted_chunk_ids: list[str],
    ) -> bool:
        cited_chunk_set = set(cited_chunk_ids)
        if cited_chunk_set & set(contradicted_chunk_ids):
            return True

        profile = run.retrieval_evidence_profile
        if profile is None:
            return False

        cited_doc_set = set(cited_doc_ids)
        for chunk_profile in profile.chunks:
            if claim_id not in chunk_profile.contradicted_claim_ids:
                continue
            if chunk_profile.chunk_id in cited_chunk_set:
                return True
            if chunk_profile.source_doc_id in cited_doc_set:
                return True
        return False

    def _claim_evidence_records(self, run: RAGRun) -> list[ClaimEvidenceRecord]:
        raw = run.metadata.get("claim_evidence_records")
        if raw is None:
            bundle = run.metadata.get("grounding_evidence_bundle")
            if isinstance(bundle, GroundingEvidenceBundle):
                raw = bundle.claim_evidence_records
            elif isinstance(bundle, dict):
                raw = bundle.get("claim_evidence_records")

        if raw is None:
            return []

        records: list[ClaimEvidenceRecord] = []
        for item in raw:
            if isinstance(item, ClaimEvidenceRecord):
                records.append(item)
            elif isinstance(item, dict):
                records.append(ClaimEvidenceRecord.model_validate(item))
        return records

    def _profile_chunk_ids_for_claim(
        self,
        profile: RetrievalEvidenceProfile | None,
        claim_id: str,
    ) -> tuple[list[str], list[str], list[str]]:
        if profile is None:
            return [], [], []

        supporting: list[str] = []
        contradicted: list[str] = []
        neutral: list[str] = []
        for chunk_profile in profile.chunks:
            if claim_id in chunk_profile.supported_claim_ids:
                supporting.append(chunk_profile.chunk_id)
            if claim_id in chunk_profile.contradicted_claim_ids:
                contradicted.append(chunk_profile.chunk_id)
            if claim_id in chunk_profile.neutral_claim_ids:
                neutral.append(chunk_profile.chunk_id)
        return supporting, contradicted, neutral

    def _ids_from_record(self, record: ClaimEvidenceRecord, *field_names: str) -> list[str]:
        values: list[str] = []
        for field_name in field_names:
            raw = getattr(record, field_name, None)
            if raw is None:
                continue
            for item in raw:
                values.append(str(item))
        return list(dict.fromkeys(values))

    def _retrieved_doc_ids(self, run: RAGRun) -> set[str]:
        return {chunk.source_doc_id for chunk in run.retrieved_chunks}

    def _is_supported_claim(self, claim: ClaimEvidenceRecord) -> bool:
        return claim.verification_label in {
            ClaimVerificationLabel.ENTAILED,
            "entailed",
        }

    def _is_unsupported_claim(self, claim: ClaimEvidenceRecord) -> bool:
        return claim.verification_label in {
            ClaimVerificationLabel.INSUFFICIENT,
            ClaimVerificationLabel.UNVERIFIED,
            "insufficient",
            "unsupported",
            "unverified",
        }

    def _risk(self, label: CitationSupportLabel) -> CitationFaithfulnessRisk:
        if label == CitationSupportLabel.FULLY_SUPPORTED:
            return CitationFaithfulnessRisk.LOW
        if label == CitationSupportLabel.PARTIALLY_SUPPORTED:
            return CitationFaithfulnessRisk.MEDIUM
        if label in {
            CitationSupportLabel.UNSUPPORTED,
            CitationSupportLabel.CITATION_PHANTOM,
            CitationSupportLabel.CITATION_MISSING,
            CitationSupportLabel.CONTRADICTED,
        }:
            return CitationFaithfulnessRisk.HIGH
        return CitationFaithfulnessRisk.UNKNOWN

    def _evidence(self, report: CitationFaithfulnessReport) -> list[str]:
        counts: dict[str, int] = {}
        for record in report.records:
            label = record.citation_support_label.value
            counts[label] = counts.get(label, 0) + 1
        count_text = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
        return [f"Citation faithfulness summary: total={len(report.records)}, {count_text}"]
