"""Analyzer for inconsistent retrieval results across sources."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


NEGATION_SIGNALS = {
    "not",
    "never",
    "no longer",
    "contrary to",
    "however",
    "but actually",
    "in fact",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
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
    "with",
}


class InconsistentChunksAnalyzer(BaseAnalyzer):
    """Detect simple contradiction signals across retrieved chunks."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        evidence: list[str] = []
        chunks = run.retrieved_chunks

        for index, left in enumerate(chunks):
            for right in chunks[index + 1 :]:
                if self._has_suspicious_negation_pair(left, right):
                    evidence.append(f"{left.chunk_id} <-> {right.chunk_id}")

        if evidence:
            return self._warn(
                FailureType.INCONSISTENT_CHUNKS,
                FailureStage.RETRIEVAL,
                evidence,
                "Review retrieved chunks for contradictory information. "
                "Consider deduplication or reranking.",
            )

        return self._pass()

    def _has_suspicious_negation_pair(
        self, left: RetrievedChunk, right: RetrievedChunk
    ) -> bool:
        left_terms = self._terms(left.text)
        right_terms = self._terms(right.text)
        shared_terms = left_terms & right_terms
        if not shared_terms:
            return False

        return self._has_nearby_negation(left.text, shared_terms) or self._has_nearby_negation(
            right.text, shared_terms
        )

    def _has_nearby_negation(self, text: str, terms: set[str]) -> bool:
        tokens = self._tokens(text)
        for index, token in enumerate(tokens):
            window = tokens[max(0, index - 5) : index + 6]
            window_text = " ".join(window)
            if token in terms and any(
                re.search(rf"\b{re.escape(signal)}\b", window_text)
                for signal in NEGATION_SIGNALS
            ):
                return True
        return False

    def _terms(self, text: str) -> set[str]:
        return {token for token in self._tokens(text) if token not in STOPWORDS}

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
