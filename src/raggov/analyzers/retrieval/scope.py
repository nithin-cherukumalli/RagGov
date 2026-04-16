"""Analyzer for retrieval scope and relevance boundaries."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class ScopeViolationAnalyzer(BaseAnalyzer):
    """Detect retrieved chunks that are off-topic relative to the query."""

    weight = 0.75

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        query_terms = self._terms(run.query)
        if not query_terms:
            return self.skip("no meaningful query terms available")

        threshold = float(self.config.get("min_overlap_ratio", 0.1))
        low_overlap: list[str] = []

        for chunk in run.retrieved_chunks:
            chunk_terms = self._terms(chunk.text)
            overlap_ratio = len(query_terms & chunk_terms) / len(query_terms)
            if overlap_ratio < threshold:
                low_overlap.append(f"{chunk.chunk_id} overlap={overlap_ratio:.2f}")

        remediation = (
            "Retrieval returned off-topic documents. Review embedding model, "
            "query preprocessing, or index quality."
        )

        if len(low_overlap) == len(run.retrieved_chunks):
            return self._fail(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                low_overlap,
                remediation,
            )
        if low_overlap:
            return self._warn(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                low_overlap,
                remediation,
            )

        return self._pass()

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }
