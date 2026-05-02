"""
Regression tests for v1 retrieval diagnosis interfaces.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from raggov.analyzers.retrieval.contradiction import (
    ContradictionLabel,
    NegationHeuristicContradictionDetector,
)
from raggov.analyzers.retrieval.freshness import AgeBasedFreshnessEvaluator
from raggov.analyzers.retrieval.inconsistency import has_suspicious_negation_pair
from raggov.analyzers.retrieval.relevance import LexicalOverlapRelevanceScorer
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import FreshnessStatus, QueryRelevanceLabel


def chunk(chunk_id: str, text: str, source_doc_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=source_doc_id,
        score=None,
    )


def corpus_entry(doc_id: str, timestamp: datetime | None) -> CorpusEntry:
    return CorpusEntry(doc_id=doc_id, text="", timestamp=timestamp)


def test_lexical_relevance_scorer_preserves_overlap_ratio_and_labels():
    scorer = LexicalOverlapRelevanceScorer(min_overlap_ratio=0.5)

    relevant = scorer.score("refund policy returns", "refund policy details")
    partial = scorer.score("refund policy returns", "refund details")
    irrelevant = scorer.score("refund policy returns", "kernel driver patch")
    unknown = scorer.score("and or the", "refund policy")

    assert relevant.label == QueryRelevanceLabel.RELEVANT
    assert relevant.score == pytest.approx(2 / 3)
    assert relevant.method == "lexical_overlap"
    assert partial.label == QueryRelevanceLabel.PARTIAL
    assert partial.score == pytest.approx(1 / 3)
    assert irrelevant.label == QueryRelevanceLabel.IRRELEVANT
    assert irrelevant.score == pytest.approx(0.0)
    assert unknown.label == QueryRelevanceLabel.UNKNOWN
    assert unknown.score is None


def test_lexical_relevance_scorer_states_it_is_not_semantic_relevance():
    score = LexicalOverlapRelevanceScorer().score("refund policy", "refund policy")
    assert score.explanation is not None
    assert "not semantic relevance" in score.explanation


def test_negation_contradiction_detector_preserves_existing_heuristic():
    left = chunk("c1", "The refund policy applies to hardware returns.")
    right = chunk("c2", "Refund policy does not apply to hardware returns.")

    result = NegationHeuristicContradictionDetector().compare_chunks(left, right)

    assert has_suspicious_negation_pair(left, right) is True
    assert result.left_id == "c1"
    assert result.right_id == "c2"
    assert result.label == ContradictionLabel.CONTRADICTION
    assert result.method == "negation_heuristic"
    assert result.explanation is not None
    assert "not NLI" in result.explanation
    assert result.score is None


def test_negation_contradiction_detector_returns_neutral_when_heuristic_is_clear():
    left = chunk("c1", "Refunds are available within 30 days.")
    right = chunk("c2", "Customers may request a refund within 30 days.")

    result = NegationHeuristicContradictionDetector().compare_chunks(left, right)

    assert has_suspicious_negation_pair(left, right) is False
    assert result.label == ContradictionLabel.NEUTRAL
    assert result.method == "negation_heuristic"


def test_age_freshness_evaluator_preserves_stale_behavior():
    now = datetime(2026, 5, 2, tzinfo=UTC)
    old = corpus_entry("doc-old", now - timedelta(days=181))
    fresh = corpus_entry("doc-fresh", now - timedelta(days=180))
    evaluator = AgeBasedFreshnessEvaluator(max_age_days=180)

    stale_result = evaluator.evaluate(chunk("c1", "old", "doc-old"), old, now)
    fresh_result = evaluator.evaluate(chunk("c2", "fresh", "doc-fresh"), fresh, now)

    assert stale_result.status == FreshnessStatus.STALE_BY_AGE
    assert stale_result.age_days == 181
    assert stale_result.method == "age_threshold"
    assert stale_result.explanation is not None
    assert "not legal or version validity" in stale_result.explanation
    assert fresh_result.status == FreshnessStatus.VALID
    assert fresh_result.age_days == 180


def test_age_freshness_evaluator_unknown_without_timestamp():
    now = datetime(2026, 5, 2, tzinfo=UTC)
    evaluator = AgeBasedFreshnessEvaluator()

    result = evaluator.evaluate(
        chunk("c1", "unknown", "doc-missing"),
        corpus_entry("doc-missing", None),
        now,
    )

    assert result.status == FreshnessStatus.UNKNOWN
    assert result.age_days is None
    assert result.warnings == [
        "Freshness cannot be determined without a corpus entry timestamp."
    ]
