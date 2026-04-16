"""Analyzer for stale or outdated retrieved context."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


class StaleRetrievalAnalyzer(BaseAnalyzer):
    """Detect retrieved chunks backed by stale corpus entries."""

    weight = 0.95

    def analyze(self, run: RAGRun) -> AnalyzerResult:
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
            )

        return self._pass()
