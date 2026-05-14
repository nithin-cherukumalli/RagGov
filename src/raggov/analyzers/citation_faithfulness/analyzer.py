"""
CitationFaithfulnessAnalyzerV0.

This analyzer builds a practical-approximation report from existing claim
grounding records and retrieval evidence. It does not run an LLM judge,
counterfactual probe, RefChecker, RAGChecker, or Wallat-style citation
faithfulness method.
"""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.evaluators.base import ExternalSignalRecord
from raggov.evaluators.citation.structured_llm import StructuredLLMCitationVerifierAdapter
from raggov.evaluators.citation.refchecker_adapter import RefCheckerCitationSignalProvider
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

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._citation_verifier = None
        self._refchecker_citation_provider: RefCheckerCitationSignalProvider | None = None
        self._external_verifier_error: str | None = None
        if self.config.get("citation_verifier") == "structured_llm":
            self._citation_verifier = StructuredLLMCitationVerifierAdapter(self.config)
            if not self._citation_verifier.is_available():
                self._external_verifier_error = (
                    "structured_llm_citation: no LLM client configured; native citation rollup fallback was used."
                )
        elif self.config.get("citation_verifier") == "refchecker":
            rc = RefCheckerCitationSignalProvider(self.config)
            if rc.is_available():
                self._refchecker_citation_provider = rc
            else:
                self._external_verifier_error = (
                    "refchecker_citation: package not installed; native citation rollup fallback was used. "
                    "Install via `pip install refchecker`."
                )

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

        if self._imprecise_doc_level_citation(run):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.CITATION_MISMATCH,
                stage=FailureStage.GROUNDING,
                evidence=evidence + [
                    "Doc-level citation spans multiple retrieved chunks; cite claim-level supporting chunks instead."
                ],
                remediation=_REMEDIATION,
                citation_faithfulness_report=report,
            )

        if report.unsupported_claim_ids and self._answer_has_explicit_citation_marker(run) and self._answer_has_specific_value(run):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.CITATION_MISMATCH,
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

        if any(
            r.citation_support_label == CitationSupportLabel.UNKNOWN
            and r.external_signal_label == "unclear"
            for r in report.records
        ):
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
            external_evaluator_used=any(r.external_signal_provider for r in built_records),
            external_evaluator_error=self._external_verifier_error,
        )

    def _build_record(
        self,
        run: RAGRun,
        claim: ClaimEvidenceRecord,
    ) -> ClaimCitationFaithfulnessRecord:
        cited_doc_ids, cited_chunk_ids, evidence_source = self._citation_sources(
            run, claim
        )
        supporting_chunk_ids = self._ids_from_record(claim, "supporting_chunk_ids")
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

        citation_record = ClaimCitationFaithfulnessRecord(
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
        return self._apply_external_signal(
            run=run,
            citation_record=citation_record,
            claim=claim,
            cited_doc_ids=cited_doc_ids,
            cited_chunk_ids=cited_chunk_ids,
        )

    def _apply_external_signal(
        self,
        *,
        run: RAGRun,
        citation_record: ClaimCitationFaithfulnessRecord,
        claim: ClaimEvidenceRecord,
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
    ) -> ClaimCitationFaithfulnessRecord:
        verifier_type = self.config.get("citation_verifier")
        if verifier_type not in ("structured_llm", "refchecker"):
            return citation_record

        citation_record.external_signal_fallback_used = True

        # --- RefChecker branch ---
        if verifier_type == "refchecker":
            if self._refchecker_citation_provider is None or self._external_verifier_error:
                citation_record.external_signal_error = self._external_verifier_error
                citation_record.warnings.append(
                    self._external_verifier_error or "refchecker_citation unavailable"
                )
                return citation_record
            cited_doc_id, cited_chunk_id, cited_text = self._primary_cited_source(
                run, cited_doc_ids, cited_chunk_ids
            )
            signals = self._refchecker_citation_provider.verify_citations(
                cited_ids=[cited_doc_id],
                chunks=[cited_text],
            )
            if not signals:
                citation_record.warnings.append("refchecker_citation returned no signal")
                return citation_record
            self._apply_signal_to_record(citation_record, signals[0])
            return citation_record

        # --- structured_llm branch ---
        if self._citation_verifier is None or self._external_verifier_error:
            citation_record.external_signal_error = self._external_verifier_error
            citation_record.warnings.append(
                self._external_verifier_error or "structured_llm_citation unavailable"
            )
            return citation_record

        cited_doc_id, cited_chunk_id, cited_text = self._primary_cited_source(
            run, cited_doc_ids, cited_chunk_ids
        )
        result = self._citation_verifier.evaluate_citation(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            cited_doc_id=cited_doc_id,
            cited_chunk_id=cited_chunk_id,
            cited_text=cited_text,
            retrieved_context=[chunk.text for chunk in run.retrieved_chunks],
        )
        if not result.succeeded or not result.signals:
            citation_record.external_signal_error = result.error
            citation_record.warnings.append(result.error or "structured_llm_citation returned no signal")
            return citation_record

        signal = result.signals[0]
        self._apply_signal_to_record(citation_record, signal)
        return citation_record

    def _primary_cited_source(
        self,
        run: RAGRun,
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
    ) -> tuple[str, str | None, str]:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in run.retrieved_chunks}
        for chunk_id in cited_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is not None:
                return chunk.source_doc_id, chunk.chunk_id, chunk.text

        cited_doc_set = set(cited_doc_ids)
        for chunk in run.retrieved_chunks:
            if chunk.source_doc_id in cited_doc_set:
                return chunk.source_doc_id, chunk.chunk_id, chunk.text

        return (
            cited_doc_ids[0] if cited_doc_ids else "",
            cited_chunk_ids[0] if cited_chunk_ids else None,
            "",
        )

    def _apply_signal_to_record(
        self,
        record: ClaimCitationFaithfulnessRecord,
        signal: ExternalSignalRecord,
    ) -> None:
        record.external_signal_fallback_used = False
        record.external_signal_provider = signal.provider.value
        record.external_signal_label = signal.label
        record.external_signal_raw_payload = signal.raw_payload
        record.explanation = signal.explanation
        label = signal.label
        if label == "supports":
            record.citation_support_label = CitationSupportLabel.FULLY_SUPPORTED
        elif label == "contradicts":
            record.citation_support_label = CitationSupportLabel.CONTRADICTED
        elif label == "does_not_support":
            record.citation_support_label = CitationSupportLabel.UNSUPPORTED
        elif label == "citation_missing":
            record.citation_support_label = CitationSupportLabel.CITATION_MISSING
        elif label == "unclear":
            record.citation_support_label = CitationSupportLabel.UNKNOWN
        record.faithfulness_risk = self._risk(record.citation_support_label)

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

    def _answer_has_explicit_citation_marker(self, run: RAGRun) -> bool:
        return "[" in run.final_answer and "]" in run.final_answer

    def _answer_has_specific_value(self, run: RAGRun) -> bool:
        import re

        return bool(re.search(r"\b\d{2,4}\b|[$₹€£]\s*\d|\b\d+(?:\.\d+)?\s*%", run.final_answer))

    def _imprecise_doc_level_citation(self, run: RAGRun) -> bool:
        if len(run.cited_doc_ids) != 1:
            return False
        cited_doc_id = run.cited_doc_ids[0]
        cited_chunks = [
            chunk for chunk in run.retrieved_chunks if chunk.source_doc_id == cited_doc_id
        ]
        if len(cited_chunks) < 2:
            return False
        if run.metadata.get("cited_chunk_ids"):
            return False
        answer_terms = {
            token
            for token in run.final_answer.lower().replace(",", " ").replace(".", " ").split()
            if len(token) > 3
        }
        chunks_with_answer_terms = 0
        for chunk in cited_chunks:
            chunk_terms = {
                token
                for token in chunk.text.lower().replace(":", " ").replace(".", " ").split()
                if len(token) > 3
            }
            if answer_terms & chunk_terms:
                chunks_with_answer_terms += 1
        return chunks_with_answer_terms >= 2

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
        evidence = [f"Citation faithfulness summary: total={len(report.records)}, {count_text}"]
        
        if report.external_evaluator_used:
            evidence.append("External citation verifier: provider=structured_llm, signal_type=citation_support")
        if report.external_evaluator_error:
            evidence.append(f"External citation verifier unavailable: {report.external_evaluator_error}")
            
        return evidence
