"""Analyzer for inconsistent retrieval results across sources."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
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

_REMEDIATION = (
    "Review retrieved chunks for contradictory information. "
    "Consider deduplication or reranking."
)


def tokens(text: str) -> list[str]:
    """Tokenize free text for inconsistency heuristics."""
    return re.findall(r"[a-z0-9]+", text.lower())


def terms(text: str) -> set[str]:
    """Extract non-stopword terms for inconsistency heuristics."""
    return {token for token in tokens(text) if token not in STOPWORDS}


def has_nearby_negation(text: str, candidate_terms: set[str]) -> bool:
    """Return whether any shared term appears near a negation signal."""
    token_list = tokens(text)
    for index, token in enumerate(token_list):
        window = token_list[max(0, index - 5) : index + 6]
        window_text = " ".join(window)
        if token in candidate_terms and any(
            re.search(rf"\b{re.escape(signal)}\b", window_text)
            for signal in NEGATION_SIGNALS
        ):
            return True
    return False


def has_suspicious_negation_pair(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    """Return whether two chunks show a contradiction-style negation pattern."""
    left_terms = terms(left.text)
    right_terms = terms(right.text)
    shared_terms = left_terms & right_terms
    if not shared_terms:
        return False

    return has_nearby_negation(left.text, shared_terms) or has_nearby_negation(
        right.text, shared_terms
    )


class InconsistentChunksAnalyzer(BaseAnalyzer):
    """Detect simple contradiction signals across retrieved chunks.

    v0 is a heuristic baseline — not calibrated, not NLI-based, not
    research-faithful (not RAGChecker / RefChecker).  Useful for early
    warning only.  Not recommended for production gating.

    When a RetrievalEvidenceProfile is attached to the run, contradictory
    chunk pairs are read from profile.contradictory_pairs (pre-computed by
    the profiler).  Otherwise, the legacy negation-pair heuristic scans all
    chunk pairs.  The analysis_source field on the returned result records
    which path was taken.
    """

    weight = 0.5

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

        if not profile.contradictory_pairs:
            return self._pass(analysis_source="retrieval_evidence_profile")

        evidence = [
            f"[profile] {left} <-> {right}"
            for left, right in profile.contradictory_pairs
        ]
        return self._warn(
            FailureType.INCONSISTENT_CHUNKS,
            FailureStage.RETRIEVAL,
            evidence,
            _REMEDIATION,
            analysis_source="retrieval_evidence_profile",
        )

    # ------------------------------------------------------------------
    # Legacy fallback (original v0 logic — preserved exactly)
    # ------------------------------------------------------------------

    def _legacy(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        evidence: list[str] = []
        chunks = run.retrieved_chunks

        for index, left in enumerate(chunks):
            for right in chunks[index + 1 :]:
                if has_suspicious_negation_pair(left, right):
                    evidence.append(f"{left.chunk_id} <-> {right.chunk_id}")

        if evidence:
            return self._warn(
                FailureType.INCONSISTENT_CHUNKS,
                FailureStage.RETRIEVAL,
                evidence,
                _REMEDIATION,
                analysis_source="legacy_heuristic_fallback",
            )

        return self._pass(analysis_source="legacy_heuristic_fallback")
