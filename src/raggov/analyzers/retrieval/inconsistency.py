"""Analyzer for inconsistent retrieval results across sources."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
from raggov.models.run import RAGRun


# Strong, polarity-bearing negations only. Discourse connectives ("however",
# "in fact", "but actually") were the dominant source of cross-topic false
# positives on multi-hop retrieval and do not by themselves indicate that two
# chunks make opposing claims, so they are excluded from contradiction signals.
NEGATION_SIGNALS = {
    "not",
    "never",
    "cannot",
    "no longer",
}

# Function words excluded when deciding whether a *shared* term is contentful.
# (Kept separate from STOPWORDS, which other modules consume via terms().)
FUNCTION_WORDS = {
    "i", "you", "he", "she", "they", "we", "him", "her", "his", "hers", "its",
    "their", "them", "our", "us", "my", "me", "your", "yours",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "these", "those", "that", "this",
    "been", "being", "am", "were", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "but", "however", "though", "although", "while", "than", "then", "also",
    "very", "more", "most", "such", "some", "any", "all", "each", "both",
    "other", "another", "one", "often", "originally",
    "into", "onto", "over", "under", "after", "before", "between", "through",
    "during", "about", "above", "below", "again", "once", "here", "there",
    "so", "because", "if", "else", "only", "own", "same", "too", "just", "now",
    "ever", "never", "not", "upon", "per", "via", "amid", "among",
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


_MULTIWORD_NEGATIONS = tuple(
    tuple(signal.split()) for signal in NEGATION_SIGNALS if " " in signal
)
_SINGLE_NEGATIONS = frozenset(
    signal for signal in NEGATION_SIGNALS if " " not in signal
)

# Tightening rationale (Task 19): the v0 rule fired on a *single* incidental
# shared token near a negation in *either* chunk, regardless of whether the two
# chunks actually disagreed — the dominant CLEAN false-positive source on
# multi-hop retrieval. A genuine contradiction is the same proposition with
# opposite polarity, so we now require (a) a strong negation whose ±5 window
# holds >=2 shared *content* terms, and (b) that same cluster asserted (present,
# negation-free) in the other chunk.
_NEGATION_WINDOW = 5
_ASSERTION_SPAN = 16
_MIN_CLUSTER = 2


def shared_content_terms(left: str, right: str) -> set[str]:
    """Shared, contentful terms between two texts (drops function/short/numeric)."""
    shared = terms(left) & terms(right)
    return {
        term
        for term in shared
        if len(term) >= 3 and term not in FUNCTION_WORDS and not term.isdigit()
    }


def _negation_positions(token_list: list[str]) -> list[int]:
    positions = [i for i, tok in enumerate(token_list) if tok in _SINGLE_NEGATIONS]
    for i in range(len(token_list)):
        for phrase in _MULTIWORD_NEGATIONS:
            if tuple(token_list[i : i + len(phrase)]) == phrase:
                positions.append(i)
    return positions


def _largest_negation_window_cluster(text: str, candidates: set[str]) -> set[str]:
    """Largest set of candidate terms co-located in one strong-negation window."""
    token_list = tokens(text)
    best: set[str] = set()
    for pos in _negation_positions(token_list):
        window = token_list[max(0, pos - _NEGATION_WINDOW) : pos + _NEGATION_WINDOW + 1]
        cluster = {tok for tok in window if tok in candidates}
        if len(cluster) > len(best):
            best = cluster
    return best


def has_nearby_negation(text: str, candidate_terms: set[str]) -> bool:
    """Return whether >=2 candidate terms share one strong-negation window."""
    return len(_largest_negation_window_cluster(text, candidate_terms)) >= _MIN_CLUSTER


def _cluster_asserted(text: str, cluster: set[str]) -> bool:
    """True if all cluster terms co-occur within a negation-free span in text."""
    token_list = tokens(text)
    negations = set(_negation_positions(token_list))
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(token_list):
        if tok in cluster:
            positions.setdefault(tok, []).append(i)
    if len(positions) < len(cluster):
        return False
    anchor = min(positions, key=lambda t: len(positions[t]))
    for start in positions[anchor]:
        lo, hi = start - _ASSERTION_SPAN, start + _ASSERTION_SPAN
        if any(lo <= i <= hi for i in negations):
            continue
        if all(any(lo <= p <= hi for p in positions[t]) for t in cluster):
            return True
    return False


def has_suspicious_negation_pair(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    """Return whether two chunks show a polarity-opposed contradiction pattern.

    Requires the same multi-term proposition to appear negated in one chunk and
    asserted in the other — not merely a shared token near a negation.
    """
    candidates = shared_content_terms(left.text, right.text)
    if len(candidates) < _MIN_CLUSTER:
        return False

    for negated, asserted in ((left.text, right.text), (right.text, left.text)):
        cluster = _largest_negation_window_cluster(negated, candidates)
        if len(cluster) >= _MIN_CLUSTER and _cluster_asserted(asserted, cluster):
            return True
    return False


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
