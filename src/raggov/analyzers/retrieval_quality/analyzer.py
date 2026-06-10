"""Narrow retrieval-stage ownership evidence source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.findings import AnalyzerFinding, AnalyzerReport
from raggov.models.run import RAGRun
from raggov.models.signals import RetrievalEvidenceMetadata


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "bring", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "list", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "why", "with",
}
_SCOPE_DRIFT_TERMS = {
    "office", "park", "parks", "public", "school", "education", "higher",
    "telangana", "ap", "jurisdiction", "state", "federal", "domestic",
    "international", "online", "store", "in-store", "current", "2024", "2025",
    "2026",
}


@dataclass(frozen=True)
class _RetrievalQualitySignal:
    signal_name: str
    failure_type: FailureType
    status: str
    severity: str
    evidence_message: str
    remediation: str
    metadata: RetrievalEvidenceMetadata


class RetrievalQualityAnalyzer(BaseAnalyzer):
    """Emit explicit retrieval-stage candidates for ownership attribution only."""

    weight = 0.98

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        signals: list[_RetrievalQualitySignal] = []
        for detector in (
            self._depth_limit_signal,
            self._scope_violation_signal,
            self._retrieval_anomaly_signal,
        ):
            signal = detector(run)
            if signal is not None:
                signals.append(signal)

        if not signals:
            return self._pass_with_report(run, "No retrieval ownership evidence detected.")

        primary = signals[0]
        metadata = [signal.metadata for signal in signals]
        findings = [
            self._finding(index=index, signal=signal)
            for index, signal in enumerate(signals, start=1)
        ]
        return AnalyzerResult(
            analyzer_name=self.name(),
            status=primary.status,  # type: ignore[arg-type]
            failure_type=primary.failure_type,
            stage=FailureStage.RETRIEVAL,
            evidence=[signal.evidence_message for signal in signals],
            signal_metadata=metadata,
            analyzer_report=AnalyzerReport(
                analyzer_name=self.name(),
                overall_status="fail" if any(s.status == "fail" for s in signals) else "warn",
                findings=findings,
                notes=["Retrieval-quality evidence is heuristic and uncalibrated."],
            ),
            remediation=primary.remediation,
        )

    def _depth_limit_signal(self, run: RAGRun) -> _RetrievalQualitySignal | None:
        expected_items = self._expected_numbered_items(run.query)
        if not expected_items:
            return None

        top_k_items = self._numbered_items(" ".join(chunk.text for chunk in run.retrieved_chunks))
        answer_items = self._numbered_items(run.final_answer)
        missing_from_context = sorted(expected_items - top_k_items)
        missing_from_answer = sorted(expected_items - answer_items)
        if not missing_from_context:
            return None

        trace = self._retrieval_trace(run)
        retrieved_k = len(run.retrieved_chunks)
        configured_top_k = self._int_value(trace, "configured_top_k", "top_k", "k") or retrieved_k
        candidate_pool_size = self._int_value(trace, "candidate_pool_size", "candidate_count", "num_candidates")
        min_required_k = self._int_value(trace, "min_required_k") or max(retrieved_k + 1, (len(expected_items) + 1) // 2)
        explicit_has_more = self._bool_value(trace, "has_more_candidates")
        if explicit_has_more is False:
            return None
        has_more_candidates = explicit_has_more
        if has_more_candidates is None and candidate_pool_size is not None:
            has_more_candidates = candidate_pool_size > retrieved_k
        if has_more_candidates is None and retrieved_k < min_required_k and missing_from_context:
            has_more_candidates = True
        if not has_more_candidates:
            return None

        relevant_hits_at_k = len(top_k_items & expected_items)
        relevant_hits_beyond_k = self._int_value(trace, "relevant_hits_beyond_k") or len(missing_from_context)
        reason = (
            "Top-k selected context omits requested requirement(s) "
            f"{missing_from_context or missing_from_answer}; retrieval depth likely truncated needed evidence."
        )
        metadata = self._metadata(
            signal_name="retrieval_depth_limit_evidence",
            evidence_ids=[chunk.chunk_id for chunk in run.retrieved_chunks],
            retrieved_k=retrieved_k,
            configured_top_k=configured_top_k,
            candidate_pool_size=candidate_pool_size,
            min_required_k=min_required_k,
            has_more_candidates=has_more_candidates,
            relevant_hits_at_k=relevant_hits_at_k,
            relevant_hits_beyond_k=relevant_hits_beyond_k,
            top_k_relevance_distribution=self._score_distribution(run),
            reason=reason,
        )
        return _RetrievalQualitySignal(
            signal_name="retrieval_depth_limit_evidence",
            failure_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
            status="fail",
            severity="high",
            evidence_message=f"[retrieval_quality:depth_limit] {reason}",
            remediation="Increase top-k or add recall/reranking evidence before grounding claims.",
            metadata=metadata,
        )

    def _scope_violation_signal(self, run: RAGRun) -> _RetrievalQualitySignal | None:
        query_terms = self._scope_terms(run.query)
        if not query_terms or not run.retrieved_chunks:
            return None
        retrieved_terms = self._scope_terms(" ".join(chunk.text for chunk in run.retrieved_chunks))
        overlap = query_terms & retrieved_terms
        missing = sorted(query_terms - retrieved_terms)
        drift_terms = sorted(retrieved_terms - query_terms)
        if not overlap or not missing or not drift_terms:
            return None
        missing_scope_terms = set(missing) & _SCOPE_DRIFT_TERMS
        drift_scope_terms = set(drift_terms) & _SCOPE_DRIFT_TERMS
        if not (missing_scope_terms and drift_scope_terms):
            return None
        if len(missing) < 1 or len(drift_terms) < 1:
            return None

        scope_match_score = len(overlap) / max(len(query_terms), 1)
        scope_drift_score = len(drift_terms) / max(len(retrieved_terms), 1)
        if scope_match_score > 0.75 or scope_drift_score < 0.25:
            return None

        reason = (
            "Retrieval returned plausible but out-of-scope evidence; "
            f"missing required scope terms {missing}, retrieved wrong-scope terms {drift_terms}."
        )
        metadata = self._metadata(
            signal_name="retrieval_scope_drift_evidence",
            evidence_ids=[chunk.chunk_id for chunk in run.retrieved_chunks],
            retrieved_k=len(run.retrieved_chunks),
            configured_top_k=self._configured_top_k(run),
            candidate_pool_size=self._candidate_pool_size(run),
            has_more_candidates=self._has_more_candidates(run),
            relevant_hits_at_k=len(overlap),
            relevant_hits_beyond_k=0,
            top_k_relevance_distribution=self._score_distribution(run),
            scope_match_score=scope_match_score,
            scope_drift_score=scope_drift_score,
            reason=reason,
            query_scope_terms=sorted(query_terms),
            retrieved_scope_terms=sorted(retrieved_terms),
            missing_required_scope_terms=missing,
            scope_drift_reason=reason,
        )
        return _RetrievalQualitySignal(
            signal_name="retrieval_scope_drift_evidence",
            failure_type=FailureType.SCOPE_VIOLATION,
            status="fail",
            severity="high",
            evidence_message=f"[retrieval_quality:scope_violation] {reason}",
            remediation="Constrain retrieval by required entity, venue, jurisdiction, time, or section scope.",
            metadata=metadata,
        )

    def _retrieval_anomaly_signal(self, run: RAGRun) -> _RetrievalQualitySignal | None:
        duplicate_ids = self._duplicate_chunk_ids(run)
        metadata_inconsistency_ids = [
            chunk.chunk_id for chunk in run.retrieved_chunks
            if chunk.metadata.get("source_trust") == "untrusted"
            or chunk.metadata.get("metadata_inconsistency") is True
        ]
        anomalous_ids = list(dict.fromkeys(duplicate_ids + metadata_inconsistency_ids))
        if not anomalous_ids:
            return None

        anomaly_type = "duplicate_chunks" if duplicate_ids else "metadata_inconsistency"
        anomaly_score = len(anomalous_ids) / max(len(run.retrieved_chunks), 1)
        reason = (
            f"Retrieval trace contains {anomaly_type}; anomalous chunks={anomalous_ids}. "
            "This is retrieval-stage evidence and does not imply security escalation."
        )
        metadata = self._metadata(
            signal_name="retrieval_anomaly_evidence",
            evidence_ids=anomalous_ids,
            retrieved_k=len(run.retrieved_chunks),
            configured_top_k=self._configured_top_k(run),
            candidate_pool_size=self._candidate_pool_size(run),
            has_more_candidates=self._has_more_candidates(run),
            relevant_hits_at_k=None,
            relevant_hits_beyond_k=None,
            top_k_relevance_distribution=self._score_distribution(run),
            retrieval_anomaly_score=anomaly_score,
            reason=reason,
            anomaly_type=anomaly_type,
            anomalous_chunk_ids=anomalous_ids,
            rank_position=self._rank_position(run, anomalous_ids[0]),
            source_trust_flags=self._source_trust_flags(run, anomalous_ids),
            metadata_inconsistency=bool(metadata_inconsistency_ids),
        )
        return _RetrievalQualitySignal(
            signal_name="retrieval_anomaly_evidence",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            status="fail",
            severity="medium",
            evidence_message=f"[retrieval_quality:retrieval_anomaly] {reason}",
            remediation="Deduplicate retrieval results and inspect source/rank metadata before context assembly.",
            metadata=metadata,
        )

    def _metadata(self, *, signal_name: str, evidence_ids: list[str], **kwargs: Any) -> RetrievalEvidenceMetadata:
        return RetrievalEvidenceMetadata(
            signal_name=signal_name,
            source_analyzer=self.name(),
            method="retrieval_depth_evidence",
            method_status="heuristic_baseline",
            calibration_status="uncalibrated",
            evidence_strength="medium",
            evidence_tier="structured",
            evidence_ids=evidence_ids,
            notes="Retrieval ownership evidence is heuristic and not calibrated.",
            **kwargs,
        )

    def _finding(self, *, index: int, signal: _RetrievalQualitySignal) -> AnalyzerFinding:
        return AnalyzerFinding(
            finding_id=f"retrieval_quality:{signal.signal_name}:{index}",
            analyzer_name=self.name(),
            failure_type=signal.failure_type,
            stage=FailureStage.RETRIEVAL,
            status=signal.status,  # type: ignore[arg-type]
            severity=signal.severity,  # type: ignore[arg-type]
            evidence_message=signal.evidence_message,
            signal_metadata=signal.metadata,
            affected_chunk_ids=signal.metadata.evidence_ids,
        )

    def _pass_with_report(self, run: RAGRun, evidence_message: str) -> AnalyzerResult:
        metadata = self._metadata(
            signal_name="retrieval_quality_no_issue_detected",
            evidence_ids=[],
            retrieved_k=len(run.retrieved_chunks),
            configured_top_k=self._configured_top_k(run),
            candidate_pool_size=self._candidate_pool_size(run),
            has_more_candidates=self._has_more_candidates(run),
            top_k_relevance_distribution=self._score_distribution(run),
            reason=evidence_message,
        )
        finding = AnalyzerFinding(
            finding_id="retrieval_quality:pass",
            analyzer_name=self.name(),
            failure_type=None,
            stage=FailureStage.RETRIEVAL,
            status="pass",
            severity="none",
            evidence_message=evidence_message,
            signal_metadata=metadata,
        )
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=[evidence_message],
            signal_metadata=[metadata],
            analyzer_report=AnalyzerReport(
                analyzer_name=self.name(),
                overall_status="pass",
                findings=[finding],
                notes=["No retrieval-stage ownership issue found by this narrow analyzer."],
            ),
        )

    def _retrieval_trace(self, run: RAGRun) -> dict[str, Any]:
        trace = run.trace or {}
        metadata = run.metadata or {}
        for source in (
            trace.get("retrieval"),
            trace.get("retrieval_trace"),
            metadata.get("retrieval"),
            metadata.get("retrieval_trace"),
            trace,
            metadata,
        ):
            if isinstance(source, dict):
                return source
        return {}

    def _configured_top_k(self, run: RAGRun) -> int | None:
        return self._int_value(self._retrieval_trace(run), "configured_top_k", "top_k", "k") or len(run.retrieved_chunks)

    def _candidate_pool_size(self, run: RAGRun) -> int | None:
        return self._int_value(self._retrieval_trace(run), "candidate_pool_size", "candidate_count", "num_candidates")

    def _has_more_candidates(self, run: RAGRun) -> bool | None:
        trace = self._retrieval_trace(run)
        explicit = self._bool_value(trace, "has_more_candidates")
        if explicit is not None:
            return explicit
        pool = self._candidate_pool_size(run)
        return pool > len(run.retrieved_chunks) if pool is not None else None

    def _int_value(self, payload: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    def _bool_value(self, payload: dict[str, Any], key: str) -> bool | None:
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return None

    def _expected_numbered_items(self, query: str) -> set[str]:
        match = re.search(r"\b(?:all\s+)?(\d+)\s+(?:requirements?|items?|steps?)\b", query.lower())
        if match is None:
            return set()
        count = int(match.group(1))
        if count <= 0 or count > 20:
            return set()
        return {str(index) for index in range(1, count + 1)}

    def _numbered_items(self, text: str) -> set[str]:
        return set(re.findall(r"\b(?:req(?:uirement)?|step)?\s*(\d{1,2})\b", text.lower()))

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())
            if token not in _STOPWORDS and len(token) > 1
        }

    def _scope_terms(self, text: str) -> set[str]:
        terms = self._terms(text)
        normalized = set(terms)
        if "dogs" in normalized:
            normalized.add("dog")
        if "parks" in normalized:
            normalized.add("park")
        return normalized

    def _score_distribution(self, run: RAGRun) -> list[float]:
        return [float(chunk.score) for chunk in run.retrieved_chunks if chunk.score is not None]

    def _duplicate_chunk_ids(self, run: RAGRun) -> list[str]:
        seen_ids: set[str] = set()
        seen_text: dict[str, str] = {}
        duplicates: list[str] = []
        for chunk in run.retrieved_chunks:
            normalized = re.sub(r"\s+", " ", chunk.text.lower()).strip()
            if chunk.chunk_id in seen_ids or normalized in seen_text:
                duplicates.append(chunk.chunk_id)
            seen_ids.add(chunk.chunk_id)
            seen_text.setdefault(normalized, chunk.chunk_id)
        return duplicates

    def _rank_position(self, run: RAGRun, chunk_id: str) -> int | None:
        for index, chunk in enumerate(run.retrieved_chunks, start=1):
            if chunk.chunk_id == chunk_id:
                return index
        return None

    def _source_trust_flags(self, run: RAGRun, chunk_ids: list[str]) -> list[str]:
        flags: list[str] = []
        wanted = set(chunk_ids)
        for chunk in run.retrieved_chunks:
            if chunk.chunk_id not in wanted:
                continue
            for key in ("source_trust", "trust_flag", "source_trust_flag"):
                value = chunk.metadata.get(key)
                if value:
                    flags.append(str(value))
        return list(dict.fromkeys(flags))
