"""
TemporalSourceValidityAnalyzerV1.

This analyzer diagnoses document-level and claim-source temporal source
validity from available lifecycle metadata and lineage fields. It does not run
an LLM judge, does not modify retrieval or citation faithfulness internals, and
is not a research-faithful VersionRAG or domain-specific temporal reasoning
implementation.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, time
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.citation_faithfulness import CitationFaithfulnessReport
from raggov.models.corpus import CorpusEntry
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.models.version_validity import (
    ClaimSourceValidityRecord,
    DocumentValidityRecord,
    DocumentValidityRisk,
    DocumentValidityStatus,
    ValidityEvidenceSource,
    VersionValidityCalibrationStatus,
    VersionValidityMethodType,
    VersionValidityReport,
)

_SKIP_REASON = "no retrieved chunks or cited documents available"
_REMEDIATION = "Review document metadata, lineage, and cited source validity."
_LIMITATIONS = [
    "v1 relies on available lifecycle metadata and document lineage fields",
    "age-based freshness is only a heuristic warning and never proves invalidity",
    "amended or revised documents may require component-level analysis",
    "domain-specific applicability belongs in optional adapters, not the core analyzer",
    "not a research-faithful VersionRAG implementation",
]
_AGE_WARNING = "age-based staleness is only a heuristic freshness warning"
_INVALID_STATUSES = {
    DocumentValidityStatus.SUPERSEDED,
    DocumentValidityStatus.WITHDRAWN,
    DocumentValidityStatus.EXPIRED,
    DocumentValidityStatus.DEPRECATED,
    DocumentValidityStatus.REPLACED,
    DocumentValidityStatus.NOT_YET_EFFECTIVE,
}
_WARN_STATUSES = {
    DocumentValidityStatus.STALE_BY_AGE,
    DocumentValidityStatus.AMENDED,
    DocumentValidityStatus.METADATA_MISSING,
    DocumentValidityStatus.APPLICABILITY_UNKNOWN,
}
_CURRENT_QUERY_TERMS = frozenset({
    "current", "latest", "today", "now", "recent", "newest", "present",
})


class TemporalSourceValidityAnalyzerV1(BaseAnalyzer):
    """Assess temporal source validity from generic lifecycle metadata."""

    weight = 0.65

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        doc_ids = self._document_ids(run)
        if not doc_ids:
            return self.skip(_SKIP_REASON)

        query_date, assumed_query_date = self._query_date(run)
        report = self._build_report(run, sorted(doc_ids), query_date, assumed_query_date)
        evidence = self._evidence(report)
        remediation = self._remediation(report)

        if report.cited_invalid_doc_ids or report.answer_bearing_invalid_doc_ids:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
                evidence=evidence,
                remediation=remediation,
                version_validity_report=report,
            )

        if report.retrieval_quality_affected_doc_ids:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
                evidence=evidence,
                remediation=remediation,
                version_validity_report=report,
            )

        if self._has_invalid_documents(report):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
                evidence=evidence,
                remediation=remediation,
                version_validity_report=report,
            )

        if self._has_warning_documents(report):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.STALE_RETRIEVAL,
                stage=FailureStage.RETRIEVAL,
                evidence=evidence,
                remediation=remediation,
                version_validity_report=report,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            version_validity_report=report,
        )

    def _build_report(
        self,
        run: RAGRun,
        doc_ids: list[str],
        query_date: datetime,
        assumed_query_date: bool,
    ) -> VersionValidityReport:
        corpus_by_doc_id = {entry.doc_id: entry for entry in run.corpus_entries}
        document_records = [
            self._document_record(
                run=run,
                doc_id=doc_id,
                corpus_entry=corpus_by_doc_id.get(doc_id),
                query_date=query_date,
                assumed_query_date=assumed_query_date,
            )
            for doc_id in doc_ids
        ]
        records_by_doc_id = {record.doc_id: record for record in document_records}
        claim_records = self._claim_records(run, records_by_doc_id)
        severity = self._severity_buckets(run, records_by_doc_id)

        return VersionValidityReport(
            run_id=run.run_id,
            query_date=query_date,
            document_records=document_records,
            claim_source_records=claim_records,
            active_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.ACTIVE),
            stale_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.STALE_BY_AGE),
            superseded_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.SUPERSEDED),
            amended_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.AMENDED),
            withdrawn_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.WITHDRAWN),
            replaced_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.REPLACED),
            deprecated_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.DEPRECATED),
            expired_doc_ids=self._ids_with_status(document_records, DocumentValidityStatus.EXPIRED),
            not_yet_effective_doc_ids=self._ids_with_status(
                document_records, DocumentValidityStatus.NOT_YET_EFFECTIVE
            ),
            metadata_missing_doc_ids=self._ids_with_status(
                document_records, DocumentValidityStatus.METADATA_MISSING
            ),
            high_risk_claim_ids=[
                record.claim_id
                for record in claim_records
                if record.claim_validity_risk == DocumentValidityRisk.HIGH
            ],
            cited_invalid_doc_ids=severity["cited_invalid_doc_ids"],
            answer_bearing_invalid_doc_ids=severity["answer_bearing_invalid_doc_ids"],
            retrieved_only_stale_doc_ids=severity["retrieved_only_stale_doc_ids"],
            stale_but_irrelevant_doc_ids=severity["stale_but_irrelevant_doc_ids"],
            retrieval_quality_affected_doc_ids=severity["retrieval_quality_affected_doc_ids"],
            retrieval_evidence_profile_used=run.retrieval_evidence_profile is not None,
            citation_faithfulness_report_used=self._citation_report(run) is not None,
            lineage_metadata_used=any(
                record.evidence_source == ValidityEvidenceSource.DOCUMENT_LINEAGE
                for record in document_records
            ),
            age_based_fallback_used=any(
                record.evidence_source == ValidityEvidenceSource.HEURISTIC_AGE_CHECK
                for record in document_records
            ),
            method_type=VersionValidityMethodType.PRACTICAL_APPROXIMATION,
            calibration_status=VersionValidityCalibrationStatus.UNCALIBRATED,
            recommended_for_gating=False,
            limitations=list(_LIMITATIONS),
        )

    def _document_record(
        self,
        *,
        run: RAGRun,
        doc_id: str,
        corpus_entry: CorpusEntry | None,
        query_date: datetime,
        assumed_query_date: bool,
    ) -> DocumentValidityRecord:
        metadata = self._metadata_for_doc(run, doc_id, corpus_entry)
        timestamp = self._timestamp_for_doc(run, doc_id, corpus_entry, metadata)
        timestamp_evidence_path = self._timestamp_evidence_path(run, doc_id, corpus_entry, metadata)
        issue_date = (
            self._parse_datetime(metadata.get("issue_date"))
            or self._parse_datetime(metadata.get("publication_date"))
            or self._parse_datetime(metadata.get("updated_at"))
            or timestamp
        )
        issue_date_evidence_path = self._first_existing_metadata_path(
            metadata, ("issue_date", "publication_date", "updated_at")
        ) or timestamp_evidence_path
        effective_date = self._first_datetime(metadata, ("effective_date", "valid_from"))
        expiry_date = self._first_datetime(metadata, ("expiry_date", "valid_to"))
        supersedes_doc_ids = self._first_list(metadata, ("supersedes", "supersedes_doc_ids"))
        superseded_by_doc_ids = self._first_list(
            metadata, ("superseded_by", "superseded_by_doc_ids")
        )
        amends_doc_ids = self._first_list(metadata, ("amends", "amends_doc_ids", "revises"))
        amended_by_doc_ids = self._first_list(
            metadata, ("amended_by", "amended_by_doc_ids", "revised_by")
        )
        replaces_doc_ids = self._first_list(metadata, ("replaces", "replaces_doc_ids"))
        replaced_by_doc_ids = self._first_list(metadata, ("replaced_by", "replaced_by_doc_ids"))
        deprecated_by_doc_ids = self._first_list(
            metadata, ("deprecated_by", "deprecated_by_doc_ids")
        )
        withdrawn_by_doc_ids = self._first_list(metadata, ("withdrawn_by", "withdrawn_by_doc_ids"))
        status = str(metadata.get("status", "")).lower()
        warnings: list[str] = []
        if assumed_query_date:
            warnings.append("query_date not provided; assumed current UTC date")

        (
            validity_status,
            risk,
            evidence_source,
            evidence_paths,
            explanation,
        ) = self._classify_document(
            status=status,
            query_date=query_date,
            timestamp=timestamp,
            issue_date=issue_date,
            issue_date_evidence_path=issue_date_evidence_path,
            effective_date=effective_date,
            expiry_date=expiry_date,
            superseded_by_doc_ids=superseded_by_doc_ids,
            amended_by_doc_ids=amended_by_doc_ids,
            replaced_by_doc_ids=replaced_by_doc_ids,
            deprecated_by_doc_ids=deprecated_by_doc_ids,
            withdrawn_by_doc_ids=withdrawn_by_doc_ids,
            metadata=metadata,
            warnings=warnings,
        )

        return DocumentValidityRecord(
            doc_id=doc_id,
            source_doc_id=doc_id,
            document_title=self._optional_str(metadata.get("document_title") or metadata.get("title")),
            document_type=self._optional_str(metadata.get("document_type")),
            department=self._optional_str(metadata.get("department")),
            version_id=self._optional_str(
                metadata.get("version_id")
                or metadata.get("version")
                or metadata.get("current_version")
            ),
            issue_date=issue_date,
            effective_date=effective_date,
            expiry_date=expiry_date,
            query_date=query_date,
            validity_status=validity_status,
            validity_risk=risk,
            supersedes_doc_ids=supersedes_doc_ids,
            superseded_by_doc_ids=superseded_by_doc_ids,
            amends_doc_ids=amends_doc_ids,
            amended_by_doc_ids=amended_by_doc_ids,
            replaces_doc_ids=replaces_doc_ids,
            replaced_by_doc_ids=replaced_by_doc_ids,
            deprecated_by_doc_ids=deprecated_by_doc_ids,
            withdrawn_by_doc_ids=withdrawn_by_doc_ids,
            evidence_source=evidence_source,
            evidence_paths=evidence_paths,
            explanation=explanation,
            warnings=warnings,
            limitations=list(_LIMITATIONS),
        )

    def _classify_document(
        self,
        *,
        status: str,
        query_date: datetime,
        timestamp: datetime | None,
        issue_date: datetime | None,
        issue_date_evidence_path: str | None,
        effective_date: datetime | None,
        expiry_date: datetime | None,
        superseded_by_doc_ids: list[str],
        amended_by_doc_ids: list[str],
        replaced_by_doc_ids: list[str],
        deprecated_by_doc_ids: list[str],
        withdrawn_by_doc_ids: list[str],
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> tuple[
        DocumentValidityStatus,
        DocumentValidityRisk,
        ValidityEvidenceSource,
        list[str],
        str,
    ]:
        if withdrawn_by_doc_ids or status == "withdrawn":
            return (
                DocumentValidityStatus.WITHDRAWN,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.DOCUMENT_LINEAGE,
                self._evidence_paths(metadata, ("withdrawn_by", "withdrawn_by_doc_ids", "status")),
                "document is withdrawn by metadata or lineage",
            )
        if superseded_by_doc_ids or status == "superseded":
            return (
                DocumentValidityStatus.SUPERSEDED,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.DOCUMENT_LINEAGE,
                self._evidence_paths(metadata, ("superseded_by", "superseded_by_doc_ids", "status")),
                "document is superseded by metadata or lineage",
            )
        if replaced_by_doc_ids or status == "replaced":
            return (
                DocumentValidityStatus.REPLACED,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.DOCUMENT_LINEAGE,
                self._evidence_paths(metadata, ("replaced_by", "replaced_by_doc_ids", "status")),
                "document is replaced by metadata or lineage",
            )
        if deprecated_by_doc_ids or status == "deprecated":
            return (
                DocumentValidityStatus.DEPRECATED,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.DOCUMENT_LINEAGE,
                self._evidence_paths(metadata, ("deprecated_by", "deprecated_by_doc_ids", "status")),
                "document is deprecated by metadata or lineage",
            )
        if expiry_date is not None and query_date > expiry_date:
            return (
                DocumentValidityStatus.EXPIRED,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.CORPUS_METADATA,
                self._evidence_paths(metadata, ("expiry_date", "valid_to")),
                "document expiry_date is before query_date",
            )
        if effective_date is not None and query_date < effective_date:
            return (
                DocumentValidityStatus.NOT_YET_EFFECTIVE,
                DocumentValidityRisk.HIGH,
                ValidityEvidenceSource.CORPUS_METADATA,
                self._evidence_paths(metadata, ("effective_date", "valid_from")),
                "document effective_date is after query_date",
            )
        if amended_by_doc_ids or status in {"amended", "revised"}:
            return (
                DocumentValidityStatus.AMENDED,
                DocumentValidityRisk.MEDIUM,
                ValidityEvidenceSource.DOCUMENT_LINEAGE,
                self._evidence_paths(
                    metadata,
                    ("amended_by", "amended_by_doc_ids", "revised_by", "status"),
                ),
                "document is amended or revised; component-level impact is unknown",
            )
        if status == "applicability_unknown":
            return (
                DocumentValidityStatus.APPLICABILITY_UNKNOWN,
                DocumentValidityRisk.UNKNOWN,
                ValidityEvidenceSource.CORPUS_METADATA,
                ["metadata.status"],
                "document applicability scope cannot be established from metadata",
            )
        if status == "draft":
            return (
                DocumentValidityStatus.APPLICABILITY_UNKNOWN,
                DocumentValidityRisk.UNKNOWN,
                ValidityEvidenceSource.CORPUS_METADATA,
                ["metadata.status"],
                "document status metadata marks it draft; publication validity is unknown",
            )
        if status in {"active", "current"}:
            return (
                DocumentValidityStatus.ACTIVE,
                DocumentValidityRisk.LOW,
                ValidityEvidenceSource.CORPUS_METADATA,
                ["metadata.status"],
                "document status metadata marks it active/current",
            )
        if status == "stale":
            return (
                DocumentValidityStatus.STALE_BY_AGE,
                DocumentValidityRisk.UNKNOWN,
                ValidityEvidenceSource.CORPUS_METADATA,
                ["metadata.status"],
                "document status metadata marks it stale",
            )
        if not self._has_useful_metadata(metadata, timestamp):
            return (
                DocumentValidityStatus.METADATA_MISSING,
                DocumentValidityRisk.UNKNOWN,
                ValidityEvidenceSource.UNAVAILABLE,
                ["metadata"],
                "no useful validity metadata available",
            )
        if not self._has_lineage_metadata(metadata) and self._is_older_than_threshold(
            query_date, issue_date or timestamp
        ):
            warnings.append(_AGE_WARNING)
            return (
                DocumentValidityStatus.STALE_BY_AGE,
                DocumentValidityRisk.UNKNOWN,
                ValidityEvidenceSource.HEURISTIC_AGE_CHECK,
                self._paths(
                    self._evidence_paths(metadata, ("issue_date", "timestamp")),
                    [issue_date_evidence_path, "config.max_age_days"],
                ),
                "document timestamp or issue_date is older than max_age_days",
            )
        return (
            DocumentValidityStatus.ACTIVE,
            DocumentValidityRisk.LOW,
            ValidityEvidenceSource.CORPUS_METADATA,
            self._evidence_paths(metadata, tuple(metadata.keys()))
            or ([issue_date_evidence_path] if issue_date_evidence_path else ["metadata"]),
            "available metadata does not indicate invalidity",
        )

    def _claim_records(
        self,
        run: RAGRun,
        records_by_doc_id: dict[str, DocumentValidityRecord],
    ) -> list[ClaimSourceValidityRecord]:
        citation_report = self._citation_report(run)
        if citation_report is None:
            return []

        claim_records: list[ClaimSourceValidityRecord] = []
        for claim in citation_report.records:
            invalid: list[str] = []
            valid: list[str] = []
            unknown: list[str] = []
            for doc_id in claim.cited_doc_ids:
                doc_record = records_by_doc_id.get(doc_id)
                if doc_record is None:
                    unknown.append(doc_id)
                elif doc_record.validity_status in _INVALID_STATUSES:
                    invalid.append(doc_id)
                elif doc_record.validity_status == DocumentValidityStatus.ACTIVE:
                    valid.append(doc_id)
                else:
                    unknown.append(doc_id)

            if invalid:
                status = records_by_doc_id[invalid[0]].validity_status
                risk = DocumentValidityRisk.HIGH
                explanation = "claim cites at least one invalid source document"
                evidence_paths = ["claim.cited_doc_ids"] + [
                    f"document_records.{doc_id}.validity_status" for doc_id in invalid
                ]
            elif claim.cited_doc_ids and len(valid) == len(claim.cited_doc_ids):
                status = DocumentValidityStatus.ACTIVE
                risk = DocumentValidityRisk.LOW
                explanation = "all cited documents are active"
                evidence_paths = ["claim.cited_doc_ids"] + [
                    f"document_records.{doc_id}.validity_status" for doc_id in valid
                ]
            elif unknown:
                status = DocumentValidityStatus.UNKNOWN
                risk = DocumentValidityRisk.UNKNOWN
                explanation = "one or more cited document validity statuses are unknown"
                evidence_paths = ["claim.cited_doc_ids"] + [
                    f"document_records.{doc_id}.validity_status" for doc_id in unknown
                ]
            else:
                status = DocumentValidityStatus.UNKNOWN
                risk = DocumentValidityRisk.UNKNOWN
                explanation = "no cited documents available for claim validity analysis"
                evidence_paths = ["claim.cited_doc_ids"]

            claim_records.append(
                ClaimSourceValidityRecord(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    cited_doc_ids=list(claim.cited_doc_ids),
                    invalid_cited_doc_ids=invalid,
                    valid_cited_doc_ids=valid,
                    unknown_validity_doc_ids=unknown,
                    claim_validity_status=status,
                    claim_validity_risk=risk,
                    does_invalid_source_affect_claim=None if invalid else False,
                    explanation=explanation,
                    evidence_paths=evidence_paths,
                )
            )
        return claim_records

    def _severity_buckets(
        self,
        run: RAGRun,
        records_by_doc_id: dict[str, DocumentValidityRecord],
    ) -> dict[str, list[str]]:
        retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
        claim_cited_doc_ids = {
            doc_id
            for record in self._claim_records(run, records_by_doc_id)
            for doc_id in record.invalid_cited_doc_ids + record.valid_cited_doc_ids + record.unknown_validity_doc_ids
        }
        cited_doc_ids = set(run.cited_doc_ids) | claim_cited_doc_ids
        current_query = self._query_requests_current_answer(run.query)

        cited_invalid_doc_ids: list[str] = []
        answer_bearing_invalid_doc_ids: list[str] = []
        retrieved_only_stale_doc_ids: list[str] = []
        stale_but_irrelevant_doc_ids: list[str] = []

        for doc_id, record in records_by_doc_id.items():
            if doc_id not in retrieved_doc_ids and doc_id not in cited_doc_ids:
                continue

            is_invalid = record.validity_status in _INVALID_STATUSES
            is_stale_by_age = record.validity_status == DocumentValidityStatus.STALE_BY_AGE
            if not is_invalid and not is_stale_by_age:
                continue

            is_cited = doc_id in cited_doc_ids
            answer_bearing = self._doc_is_answer_bearing(run, doc_id)
            query_relevant = self._doc_is_query_relevant(run, doc_id)

            if is_invalid and is_cited:
                cited_invalid_doc_ids.append(doc_id)
                continue

            if (is_invalid or (is_stale_by_age and current_query)) and answer_bearing:
                answer_bearing_invalid_doc_ids.append(doc_id)
                continue

            if is_stale_by_age and doc_id in retrieved_doc_ids and not is_cited:
                if query_relevant:
                    retrieved_only_stale_doc_ids.append(doc_id)
                elif self._has_current_alternative(run, doc_id):
                    stale_but_irrelevant_doc_ids.append(doc_id)
                else:
                    retrieved_only_stale_doc_ids.append(doc_id)

        retrieval_quality_affected_doc_ids = []
        if retrieved_only_stale_doc_ids and self._retrieval_quality_affected(run, retrieved_only_stale_doc_ids):
            retrieval_quality_affected_doc_ids = list(retrieved_only_stale_doc_ids)

        return {
            "cited_invalid_doc_ids": sorted(dict.fromkeys(cited_invalid_doc_ids)),
            "answer_bearing_invalid_doc_ids": sorted(dict.fromkeys(answer_bearing_invalid_doc_ids)),
            "retrieved_only_stale_doc_ids": sorted(dict.fromkeys(retrieved_only_stale_doc_ids)),
            "stale_but_irrelevant_doc_ids": sorted(dict.fromkeys(stale_but_irrelevant_doc_ids)),
            "retrieval_quality_affected_doc_ids": sorted(dict.fromkeys(retrieval_quality_affected_doc_ids)),
        }

    def _document_ids(self, run: RAGRun) -> set[str]:
        return {chunk.source_doc_id for chunk in run.retrieved_chunks} | set(run.cited_doc_ids)

    def _query_requests_current_answer(self, query: str) -> bool:
        tokens = self._terms(query)
        return bool(tokens & _CURRENT_QUERY_TERMS)

    def _doc_text(self, run: RAGRun, doc_id: str) -> str:
        return " ".join(chunk.text for chunk in run.retrieved_chunks if chunk.source_doc_id == doc_id)

    def _doc_is_answer_bearing(self, run: RAGRun, doc_id: str) -> bool:
        if doc_id in run.cited_doc_ids:
            return True

        doc_text = self._doc_text(run, doc_id)
        if not doc_text:
            return False

        doc_terms = self._terms(doc_text)
        answer_terms = self._terms(run.final_answer)
        overlap = doc_terms & answer_terms
        if len(overlap) >= 2:
            return True

        answer_signals = [
            signal
            for signal in re.findall(r"\b[A-Z][a-zA-Z0-9_.-]+\b|\b\d+(?:\.\d+)?%?\b", run.final_answer)
            if signal.isdigit()
            or any(char.islower() for char in signal)
        ]
        lowered_doc = doc_text.lower()
        return any(signal.lower() in lowered_doc for signal in answer_signals)

    def _doc_is_query_relevant(self, run: RAGRun, doc_id: str) -> bool:
        doc_terms = self._terms(self._doc_text(run, doc_id))
        if not doc_terms:
            return False
        return bool(doc_terms & self._terms(run.query))

    def _retrieval_quality_affected(self, run: RAGRun, stale_doc_ids: list[str]) -> bool:
        retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
        if len(retrieved_doc_ids) <= 1:
            return False
        if run.cited_doc_ids:
            return any(doc_id not in run.cited_doc_ids for doc_id in stale_doc_ids)
        return True

    def _has_current_alternative(self, run: RAGRun, stale_doc_id: str) -> bool:
        for chunk in run.retrieved_chunks:
            if chunk.source_doc_id == stale_doc_id:
                continue
            if self._doc_is_answer_bearing(run, chunk.source_doc_id):
                return True
        return bool(run.cited_doc_ids)

    def _query_date(self, run: RAGRun) -> tuple[datetime, bool]:
        raw = run.metadata.get("query_date")
        if raw is not None:
            parsed = self._parse_datetime(raw)
            if parsed is not None:
                return parsed, False
        return datetime.now(UTC), True

    def _metadata_for_doc(
        self,
        run: RAGRun,
        doc_id: str,
        corpus_entry: CorpusEntry | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if corpus_entry is not None:
            metadata.update(corpus_entry.metadata)
        for chunk in run.retrieved_chunks:
            if chunk.source_doc_id == doc_id:
                metadata.update(chunk.metadata)
                metadata.update(self._text_lifecycle_metadata(chunk.text))
        return metadata

    def _timestamp_for_doc(
        self,
        run: RAGRun,
        doc_id: str,
        corpus_entry: CorpusEntry | None,
        metadata: dict[str, Any],
    ) -> datetime | None:
        if corpus_entry is not None and corpus_entry.timestamp is not None:
            return self._ensure_aware(corpus_entry.timestamp)
        parsed = self._parse_datetime(metadata.get("timestamp"))
        if parsed is not None:
            return parsed
        for field in ("updated_at", "publication_date"):
            parsed = self._parse_datetime(metadata.get(field))
            if parsed is not None:
                return parsed
        for chunk in run.retrieved_chunks:
            if chunk.source_doc_id == doc_id:
                parsed = self._parse_datetime(chunk.metadata.get("timestamp"))
                if parsed is not None:
                    return parsed
        return None

    def _timestamp_evidence_path(
        self,
        run: RAGRun,
        doc_id: str,
        corpus_entry: CorpusEntry | None,
        metadata: dict[str, Any],
    ) -> str | None:
        if corpus_entry is not None and corpus_entry.timestamp is not None:
            return "corpus_entries.timestamp"
        if self._parse_datetime(metadata.get("timestamp")) is not None:
            return "metadata.timestamp"
        for chunk in run.retrieved_chunks:
            if (
                chunk.source_doc_id == doc_id
                and self._parse_datetime(chunk.metadata.get("timestamp")) is not None
            ):
                return "retrieved_chunks.metadata.timestamp"
        return None

    def _citation_report(self, run: RAGRun) -> CitationFaithfulnessReport | None:
        if run.citation_faithfulness_report is not None:
            return run.citation_faithfulness_report
        raw = run.metadata.get("citation_faithfulness_report")
        if isinstance(raw, CitationFaithfulnessReport):
            return raw
        if isinstance(raw, dict):
            return CitationFaithfulnessReport.model_validate(raw)
        return None

    def _parse_datetime(self, raw: Any) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return self._ensure_aware(raw)
        if isinstance(raw, str):
            value = raw.strip()
            if not value:
                return None
            if value.endswith("Z"):
                value = f"{value[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None and parsed.time() == time.min:
                parsed = datetime.combine(parsed.date(), time.min, tzinfo=UTC)
            return self._ensure_aware(parsed)
        return None

    def _ensure_aware(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _list(self, raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw] if raw else []
        return [str(item) for item in raw]

    def _first_datetime(self, metadata: dict[str, Any], fields: tuple[str, ...]) -> datetime | None:
        for field in fields:
            parsed = self._parse_datetime(metadata.get(field))
            if parsed is not None:
                return parsed
        return None

    def _first_existing_metadata_path(self, metadata: dict[str, Any], fields: tuple[str, ...]) -> str | None:
        for field in fields:
            if metadata.get(field):
                return f"metadata.{field}"
        return None

    def _first_list(self, metadata: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
        for field in fields:
            values = self._list(metadata.get(field))
            if values:
                return values
        return []

    def _evidence_paths(self, metadata: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
        return [f"metadata.{field}" for field in fields if metadata.get(field)]

    def _paths(self, *path_groups: list[str | None]) -> list[str]:
        paths: list[str] = []
        for group in path_groups:
            for path in group:
                if path is not None and path not in paths:
                    paths.append(path)
        return paths

    def _optional_str(self, raw: Any) -> str | None:
        if raw is None:
            return None
        value = str(raw)
        return value if value else None

    def _has_lineage_metadata(self, metadata: dict[str, Any]) -> bool:
        return any(
            metadata.get(field)
            for field in (
                "supersedes_doc_ids",
                "superseded_by_doc_ids",
                "supersedes",
                "superseded_by",
                "amends_doc_ids",
                "amended_by_doc_ids",
                "amends",
                "amended_by",
                "revises",
                "revised_by",
                "replaces",
                "replaced_by",
                "replaces_doc_ids",
                "replaced_by_doc_ids",
                "deprecated_by",
                "deprecated_by_doc_ids",
                "withdrawn_by",
                "withdrawn_by_doc_ids",
                "status",
                "effective_date",
                "expiry_date",
                "valid_from",
                "valid_to",
                "version_id",
            )
        )

    def _has_useful_metadata(
        self, metadata: dict[str, Any], timestamp: datetime | None
    ) -> bool:
        return bool(metadata) or timestamp is not None

    def _is_older_than_threshold(
        self, query_date: datetime, date_value: datetime | None
    ) -> bool:
        if date_value is None:
            return False
        max_age_days = int(self.config.get("max_age_days", 180))
        return (query_date - date_value).days > max_age_days

    def _ids_with_status(
        self,
        records: list[DocumentValidityRecord],
        status: DocumentValidityStatus,
    ) -> list[str]:
        return [record.doc_id for record in records if record.validity_status == status]

    def _has_invalid_documents(self, report: VersionValidityReport) -> bool:
        return any(
            [
                report.superseded_doc_ids,
                report.withdrawn_doc_ids,
                report.replaced_doc_ids,
                report.deprecated_doc_ids,
                report.expired_doc_ids,
                report.not_yet_effective_doc_ids,
            ]
        )

    def _has_warning_documents(self, report: VersionValidityReport) -> bool:
        return any(
            record.validity_status in _WARN_STATUSES
            for record in report.document_records
        ) and not (
            report.stale_but_irrelevant_doc_ids
            and not report.retrieved_only_stale_doc_ids
            and not report.answer_bearing_invalid_doc_ids
        )

    def _evidence(self, report: VersionValidityReport) -> list[str]:
        return [
            "Temporal source validity summary: "
            f"active={len(report.active_doc_ids)}, "
            f"stale={len(report.stale_doc_ids)}, "
            f"superseded={len(report.superseded_doc_ids)}, "
            f"amended={len(report.amended_doc_ids)}, "
            f"withdrawn={len(report.withdrawn_doc_ids)}, "
            f"replaced={len(report.replaced_doc_ids)}, "
            f"deprecated={len(report.deprecated_doc_ids)}, "
            f"expired={len(report.expired_doc_ids)}, "
            f"not_yet_effective={len(report.not_yet_effective_doc_ids)}, "
            f"metadata_missing={len(report.metadata_missing_doc_ids)}; "
            f"cited_invalid={len(report.cited_invalid_doc_ids)}, "
            f"answer_bearing_invalid={len(report.answer_bearing_invalid_doc_ids)}, "
            f"retrieved_only_stale={len(report.retrieved_only_stale_doc_ids)}, "
            f"stale_irrelevant={len(report.stale_but_irrelevant_doc_ids)}, "
            f"retrieval_quality_affected={len(report.retrieval_quality_affected_doc_ids)}"
        ]

    def _remediation(self, report: VersionValidityReport) -> str:
        if report.cited_invalid_doc_ids:
            primary_doc_id = report.cited_invalid_doc_ids[0]
            status = self._status_for_doc(report, primary_doc_id)
            if status == DocumentValidityStatus.EXPIRED:
                return (
                    "Lifecycle management: cited expired sources should be filtered or "
                    "replaced with currently valid alternatives."
                )
            if status == DocumentValidityStatus.WITHDRAWN:
                return (
                    "Quarantine stale docs: cited withdrawn sources should be blocked "
                    "from answer generation and retrieval results."
                )
            if status == DocumentValidityStatus.SUPERSEDED:
                return (
                    "Version control check: cited superseded sources should be replaced "
                    "with their current versions."
                )
            if status == DocumentValidityStatus.NOT_YET_EFFECTIVE:
                return (
                    "Temporal filtering: exclude not-yet-effective sources when the "
                    "query asks about the current state."
                )
            return _REMEDIATION

        if report.answer_bearing_invalid_doc_ids:
            return (
                "Lifecycle management: answer-bearing invalid sources should not drive "
                "claim generation or support judgments."
            )

        if report.retrieval_quality_affected_doc_ids:
            return (
                "Retrieval deduplication: stale retrieved sources are polluting ranking "
                "or provenance and should be demoted or removed."
            )

        return _REMEDIATION

    def _status_for_doc(
        self,
        report: VersionValidityReport,
        doc_id: str,
    ) -> DocumentValidityStatus | None:
        for record in report.document_records:
            if record.doc_id == doc_id:
                return record.validity_status
        return None

    def _text_lifecycle_metadata(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        metadata: dict[str, Any] = {}
        if "superseded by" in lowered:
            metadata["status"] = "superseded"
            match = re.search(r"superseded\s+by\s+([A-Za-z0-9_.-]+)", text, re.IGNORECASE)
            if match:
                metadata["superseded_by"] = match.group(1)
        elif "withdrawn" in lowered:
            metadata["status"] = "withdrawn"
            metadata["withdrawn_by"] = "text_lifecycle_marker"
        elif "deprecated" in lowered:
            metadata["status"] = "deprecated"
            metadata["deprecated_by"] = "text_lifecycle_marker"
        elif "replaced by" in lowered:
            metadata["status"] = "replaced"
            match = re.search(r"replaced\s+by\s+([A-Za-z0-9_.-]+)", text, re.IGNORECASE)
            if match:
                metadata["replaced_by"] = match.group(1)

        expired = re.search(r"\bexpired\s+(\d{4})\b", text, re.IGNORECASE)
        if expired:
            metadata["expiry_date"] = f"{expired.group(1)}-12-31"
        effective = re.search(r"\beffective\s+(\d{4})\b", text, re.IGNORECASE)
        if effective:
            metadata["effective_date"] = f"{effective.group(1)}-01-01"
        as_of = re.search(r"\bas\s+of\s+(\d{4})\b|\((\d{4})\)", text, re.IGNORECASE)
        if as_of:
            year = as_of.group(1) or as_of.group(2)
            metadata.setdefault("issue_date", f"{year}-01-01")
        return metadata

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }


class VersionValidityAnalyzerV1(TemporalSourceValidityAnalyzerV1):
    """Backward-compatible name for the temporal source validity analyzer."""
