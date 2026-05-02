"""
v1 interface and baseline implementation for retrieval freshness evaluation.

IMPORTANT:
- This module defines the evaluation interface only.
- AgeBasedFreshnessEvaluator is NOT legal or version validity checking.
  It compares a document timestamp against a calendar age threshold only.
- A document within the age threshold may still be superseded by a policy
  amendment, legal update, or government notification.
- Future implementations may use supersession metadata or legal date ranges;
  they are not provided here and must not be assumed to be available.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import FreshnessStatus

# Matches StaleRetrievalAnalyzer and RetrievalEvidenceProfilerV0 defaults exactly.
_DEFAULT_MAX_AGE_DAYS: int = 180


class FreshnessEvaluation(BaseModel):
    """
    Output of a single freshness evaluation call.

    This is a structured intermediate result, not a final gating signal.
    The status and age_days are method-dependent; do not compare across
    different FreshnessEvaluator implementations without normalisation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: FreshnessStatus = FreshnessStatus.UNKNOWN
    age_days: Optional[int] = None
    method: str = "unknown"
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


@runtime_checkable
class FreshnessEvaluator(Protocol):
    """
    Protocol for chunk freshness evaluation.

    Implementations must return a FreshnessEvaluation for any (chunk,
    corpus_entry, now) triple without raising.  corpus_entry may be None
    when no metadata is available for the chunk's source document.
    """

    def evaluate(
        self,
        chunk: RetrievedChunk,
        corpus_entry: Optional[CorpusEntry],
        now: datetime,
    ) -> FreshnessEvaluation:
        """Evaluate the freshness of the source document backing the chunk."""
        ...


class AgeBasedFreshnessEvaluator:
    """
    Freshness evaluator using calendar age against a configurable threshold.

    NOT legal or version validity checking.  Marks documents older than
    max_age_days as stale_by_age; documents within the threshold as valid.

    Behaviour is identical to the v0 age heuristic in
    StaleRetrievalAnalyzer and RetrievalEvidenceProfilerV0.

    Limitations:
    - A document published yesterday may be superseded by a later amendment.
    - A document older than the threshold may still be authoritative if
      no newer version exists.
    - Timezone-naive timestamps are assumed to be UTC.
    """

    def __init__(self, max_age_days: int = _DEFAULT_MAX_AGE_DAYS) -> None:
        self._max_age_days = max_age_days

    def evaluate(
        self,
        chunk: RetrievedChunk,
        corpus_entry: Optional[CorpusEntry],
        now: datetime,
    ) -> FreshnessEvaluation:
        if corpus_entry is None or corpus_entry.timestamp is None:
            return FreshnessEvaluation(
                status=FreshnessStatus.UNKNOWN,
                age_days=None,
                method="age_threshold",
                explanation=(
                    "No corpus entry or timestamp available for this chunk. "
                    "This age-based method is not legal or version validity checking."
                ),
                warnings=["Freshness cannot be determined without a corpus entry timestamp."],
            )

        ts = corpus_entry.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        age_days = (now - ts).days
        is_stale = ts < (now - timedelta(days=self._max_age_days))

        return FreshnessEvaluation(
            status=FreshnessStatus.STALE_BY_AGE if is_stale else FreshnessStatus.VALID,
            age_days=age_days,
            method="age_threshold",
            explanation=(
                "This age-based method is not legal or version validity checking. "
                f"Document is {age_days} days old "
                f"(threshold: {self._max_age_days} days, "
                f"status: {'stale' if is_stale else 'valid'})."
            ),
        )
