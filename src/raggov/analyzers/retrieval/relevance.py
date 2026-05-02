"""
v1 interface and baseline implementation for retrieval relevance scoring.

IMPORTANT:
- This module defines the scoring interface only.
- LexicalOverlapRelevanceScorer is NOT semantic relevance.
  It measures exact token intersection between query and chunk text.
- A high overlap score does not imply the chunk is substantively useful.
- Future implementations may use embeddings or LLM judges; they are not
  provided here and must not be assumed to be available.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from raggov.models.retrieval_evidence import QueryRelevanceLabel, RelevanceMethod

RetrievalRelevanceMethod = Literal[
    "lexical_overlap",
    "embedding_similarity",
    "llm_judge",
    "unavailable",
]

# Matches ScopeViolationAnalyzer and RetrievalEvidenceProfilerV0 defaults exactly.
_DEFAULT_MIN_OVERLAP_RATIO: float = 0.1

_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


class RetrievalRelevanceScore(BaseModel):
    """
    Output of a single relevance scoring call.

    This is a structured intermediate result, not a final gating signal.
    The score and label are method-dependent and must not be compared across
    different RelevanceMethods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: Optional[float] = None
    label: QueryRelevanceLabel = QueryRelevanceLabel.UNKNOWN
    method: RetrievalRelevanceMethod = "unavailable"
    explanation: Optional[str] = None
    error: Optional[str] = None


@runtime_checkable
class RetrievalRelevanceScorer(Protocol):
    """
    Protocol for query-to-chunk relevance scoring.

    Implementations must return a RetrievalRelevanceScore for any (query,
    chunk_text) pair without raising.  Errors must be captured in the
    result's error field rather than propagated.
    """

    def score(self, query: str, chunk_text: str) -> RetrievalRelevanceScore:
        """Score the relevance of chunk_text to query."""
        ...


class LexicalOverlapRelevanceScorer:
    """
    Relevance scorer based on exact token overlap.

    NOT semantic relevance.  Computes the fraction of non-stopword query
    tokens that appear verbatim in the chunk text after lowercasing.

    Behaviour is identical to the v0 lexical heuristic in
    ScopeViolationAnalyzer and RetrievalEvidenceProfilerV0.

    Limitations:
    - Stemming and lemmatisation are not applied ("refund" ≠ "refunds").
    - Token order is ignored.
    - Semantic synonyms are not matched.
    - A zero-overlap chunk may still be substantively relevant.
    """

    def __init__(self, min_overlap_ratio: float = _DEFAULT_MIN_OVERLAP_RATIO) -> None:
        self._threshold = min_overlap_ratio

    def score(self, query: str, chunk_text: str) -> RetrievalRelevanceScore:
        query_terms = _terms(query)
        if not query_terms:
            return RetrievalRelevanceScore(
                score=None,
                label=QueryRelevanceLabel.UNKNOWN,
                method=RelevanceMethod.LEXICAL_OVERLAP.value,
                explanation=(
                    "No meaningful query terms after stopword removal. "
                    "This lexical-overlap method is not semantic relevance."
                ),
            )
        chunk_terms = _terms(chunk_text)
        ratio = len(query_terms & chunk_terms) / len(query_terms)
        if ratio >= self._threshold:
            label = QueryRelevanceLabel.RELEVANT
        elif ratio > 0.0:
            label = QueryRelevanceLabel.PARTIAL
        else:
            label = QueryRelevanceLabel.IRRELEVANT
        return RetrievalRelevanceScore(
            score=ratio,
            label=label,
            method=RelevanceMethod.LEXICAL_OVERLAP.value,
            explanation=(
                "This lexical-overlap method is not semantic relevance. "
                f"Overlap ratio {ratio:.3f} against threshold {self._threshold:.3f}. "
                f"Query terms: {len(query_terms)}, matched: {len(query_terms & chunk_terms)}."
            ),
        )
