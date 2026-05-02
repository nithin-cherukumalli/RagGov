"""Analyzer for stale or outdated retrieved context."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
from raggov.models.run import RAGRun


class StaleRetrievalAnalyzer(BaseAnalyzer):
    """Detect retrieved chunks backed by stale corpus entries.

    v0 is a heuristic baseline — not calibrated, not NLI-based, not
    research-faithful (not RAGChecker / RefChecker).  Useful for early
    warning only.  Not recommended for production gating.

    When a RetrievalEvidenceProfile is attached to the run, stale document
    IDs are read from profile.stale_doc_ids (pre-computed by the profiler).
    Otherwise, the legacy age-threshold check against corpus entry timestamps
    is used.  The analysis_source field on the returned result records which
    path was taken.
    """

    weight = 0.95

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        profile: RetrievalEvidenceProfile | None = run.retrieval_evidence_profile
        if profile is not None:
            return self._from_profile(run, profile)
        return self._legacy(run)

    # ------------------------------------------------------------------
    # Profile path
    # ------------------------------------------------------------------

    def _from_profile(
        self, run: RAGRun, profile: RetrievalEvidenceProfile
    ) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        if not profile.stale_doc_ids:
            return self._pass(analysis_source="retrieval_evidence_profile")

        max_age_days = int(self.config.get("max_age_days", 180))
        evidence = [
            f"[profile] stale document: {doc_id}"
            for doc_id in profile.stale_doc_ids
        ]
        return self._fail(
            FailureType.STALE_RETRIEVAL,
            FailureStage.RETRIEVAL,
            evidence,
            f"Re-index documents older than {max_age_days} days or add freshness "
            "filtering to retrieval.",
            analysis_source="retrieval_evidence_profile",
        )

    # ------------------------------------------------------------------
    # Legacy fallback (original v0 logic — preserved exactly)
    # ------------------------------------------------------------------

    def _legacy(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")
        if not run.corpus_entries:
            return self.skip("no corpus metadata available")

        max_age_days = int(self.config.get("max_age_days", 180))
        stale_before = datetime.now(UTC) - timedelta(days=max_age_days)
        corpus_by_doc_id = {entry.doc_id: entry for entry in run.corpus_entries}
        evidence: list[str] = []

        for chunk in run.retrieved_chunks:
            entry = corpus_by_doc_id.get(chunk.source_doc_id)
            if entry is None or entry.timestamp is None:
                continue

            timestamp = entry.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            if timestamp < stale_before:
                age_days = (datetime.now(UTC) - timestamp).days
                evidence.append(f"{entry.doc_id} is {age_days} days old")

        if evidence:
            return self._fail(
                FailureType.STALE_RETRIEVAL,
                FailureStage.RETRIEVAL,
                evidence,
                "Re-index documents older than "
                f"{max_age_days} days or add freshness filtering to retrieval.",
                analysis_source="legacy_heuristic_fallback",
            )

        return self._pass(analysis_source="legacy_heuristic_fallback")
