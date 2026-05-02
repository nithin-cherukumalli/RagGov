"""
v1 interface and baseline implementation for chunk contradiction detection.

IMPORTANT:
- This module defines the detection interface only.
- NegationHeuristicContradictionDetector is NOT NLI (Natural Language Inference).
  It uses token-window negation signals, not entailment models.
- A contradiction label from this detector does not prove semantic opposition.
- Future implementations may use NLI or LLM judges; they are not provided
  here and must not be assumed to be available.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from raggov.analyzers.retrieval.inconsistency import has_suspicious_negation_pair
from raggov.models.chunk import RetrievedChunk


class ContradictionLabel(str, Enum):
    """Contradiction relationship between two retrieved chunks."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"


class ContradictionResult(BaseModel):
    """
    Output of a single chunk-pair contradiction detection call.

    This is a structured intermediate result, not a final gating signal.
    The label and score are method-dependent and must not be compared
    across different detection methods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_id: str
    right_id: str
    label: ContradictionLabel = ContradictionLabel.UNKNOWN
    score: Optional[float] = None
    method: str = "unknown"
    explanation: Optional[str] = None
    error: Optional[str] = None


@runtime_checkable
class ContradictionDetector(Protocol):
    """
    Protocol for pairwise chunk contradiction detection.

    Implementations must return a ContradictionResult for any (left, right)
    chunk pair without raising.  Errors must be captured in the result's
    error field rather than propagated.
    """

    def compare_chunks(
        self, left: RetrievedChunk, right: RetrievedChunk
    ) -> ContradictionResult:
        """Compare two chunks and return a contradiction result."""
        ...


class NegationHeuristicContradictionDetector:
    """
    Contradiction detector using negation token-window signals.

    NOT NLI.  Checks whether terms shared by both chunks appear near negation
    signals ("not", "never", "no longer", etc.) within a ±5 token window.

    Behaviour is identical to the v0 negation heuristic in
    InconsistentChunksAnalyzer and RetrievalEvidenceProfilerV0.

    Limitations:
    - Does not model semantic entailment or contradiction.
    - Negation signals far from shared terms are missed.
    - Implicit contradictions (no negation word present) are missed.
    - False positives occur when negation is scoped to unshared terms.
    """

    def compare_chunks(
        self, left: RetrievedChunk, right: RetrievedChunk
    ) -> ContradictionResult:
        try:
            flagged = has_suspicious_negation_pair(left, right)
        except Exception as exc:
            return ContradictionResult(
                left_id=left.chunk_id,
                right_id=right.chunk_id,
                label=ContradictionLabel.UNKNOWN,
                method="negation_heuristic",
                error=str(exc),
            )

        if flagged:
            return ContradictionResult(
                left_id=left.chunk_id,
                right_id=right.chunk_id,
                label=ContradictionLabel.CONTRADICTION,
                method="negation_heuristic",
                explanation=(
                    "This negation heuristic is not NLI. Shared terms were detected "
                    "near a negation signal in a ±5 token window."
                ),
            )
        return ContradictionResult(
            left_id=left.chunk_id,
            right_id=right.chunk_id,
            label=ContradictionLabel.NEUTRAL,
            method="negation_heuristic",
            explanation=(
                "This negation heuristic is not NLI. No negation pattern was "
                "detected near shared terms."
            ),
        )
