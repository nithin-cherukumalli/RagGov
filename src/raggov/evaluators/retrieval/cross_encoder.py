"""Cross-encoder retrieval relevance signal provider.

sentence-transformers is an optional lazy dependency.
Install: pip install sentence-transformers

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (configurable).
All outputs are uncalibrated; do not use for gating without local calibration.

Config keys:
  model_name          str    Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k               int    Only score top-k chunks (default: all)
  batch_size          int    Inference batch size (default: 32)
  relevant_threshold  float  Score >= this → "relevant" (default: 3.0, uncalibrated)
  partial_threshold   float  Score >= this → "partial"  (default: 0.0, uncalibrated)
  # Scores below partial_threshold → "irrelevant"
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Any

from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.models.run import RAGRun

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_RELEVANT_THRESHOLD = 3.0   # raw logit — uncalibrated default
_DEFAULT_PARTIAL_THRESHOLD = 0.0    # raw logit — uncalibrated default
_DEFAULT_BATCH_SIZE = 32

_BASE_LIMITATIONS = [
    "uncalibrated_locally: thresholds are heuristic defaults, not calibrated on GovRAG labeled data",
    "recommended_for_gating=False: do not gate on these scores without local calibration",
    "cross-encoder scores are model-dependent; not comparable across different models",
    "sentence-transformers is an optional dependency; not available in all environments",
]


class CrossEncoderRetrievalRelevanceProvider:
    """Score query–chunk relevance using a cross-encoder model.

    Returns one ExternalSignalRecord per chunk with:
      - provider=cross_encoder
      - signal_type=retrieval_relevance
      - metric_name="cross_encoder_relevance"
      - value=float relevance score
      - label="relevant" | "partial" | "irrelevant"
      - affected_chunk_ids=[chunk_id]
      - affected_doc_ids=[source_doc_id]
      - calibration_status="uncalibrated_locally"
      - recommended_for_gating=False

    Integration:
      Apply signals to RetrievalEvidenceProfile via
      apply_external_relevance_signals() in evidence_profile.py.
    """

    name: str = "cross_encoder_relevance"
    provider: ExternalEvaluatorProvider = ExternalEvaluatorProvider.cross_encoder

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._model_name: str = cfg.get("model_name", _DEFAULT_MODEL_NAME)
        self._top_k: int | None = cfg.get("top_k")
        self._batch_size: int = int(cfg.get("batch_size", _DEFAULT_BATCH_SIZE))
        self._relevant_threshold: float = float(
            cfg.get("relevant_threshold", _DEFAULT_RELEVANT_THRESHOLD)
        )
        self._partial_threshold: float = float(
            cfg.get("partial_threshold", _DEFAULT_PARTIAL_THRESHOLD)
        )
        self._allow_model_downloads: bool = bool(cfg.get("allow_model_downloads", False))
        self._model: Any | None = None
        self._single_score_cursor: int = 0

    # ------------------------------------------------------------------
    # ExternalSignalProvider protocol
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return (
            self._model is not None
            or "sentence_transformers" in sys.modules
            or importlib.util.find_spec("sentence_transformers") is not None
        )

    def evaluate(self, run: RAGRun) -> ExternalEvaluationResult:
        """Score all retrieved chunks in run against the query.

        Returns one ExternalSignalRecord per chunk.
        Missing dependency returns succeeded=False, missing_dependency=True.
        No silent fallback: errors are always surfaced.
        """
        if not self.is_available():
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                missing_dependency=True,
                error=(
                    "cross_encoder_relevance: sentence-transformers not installed. "
                    "Run `pip install sentence-transformers` to enable this adapter. "
                    "Native lexical-overlap fallback remains active in RetrievalEvidenceProfilerV0."
                ),
            )
        if not run.retrieved_chunks:
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=True,
                signals=[],
            )
        try:
            chunks = (
                run.retrieved_chunks[: self._top_k]
                if self._top_k is not None
                else run.retrieved_chunks
            )
            signals = self._score_chunks(run.query, chunks)
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=True,
                signals=signals,
                raw_payload={"model_name": self._model_name, "chunk_count": len(signals)},
            )
        except Exception as exc:
            logger.warning("cross_encoder_relevance evaluation failed: %s", exc)
            return ExternalEvaluationResult(
                provider=self.provider,
                succeeded=False,
                error=f"cross_encoder_relevance: evaluation failed: {exc}",
            )

    def score_relevance(
        self,
        query: str,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        doc_ids: list[str | None] | None = None,
    ) -> list[ExternalSignalRecord]:
        """Score raw text chunks against query.

        chunk_ids and doc_ids are optional metadata attached to each signal.
        Returns [] (not raises) when adapter is unavailable — callers must
        check is_available() before calling if they need to distinguish
        unavailability from empty results.
        """
        if not self.is_available():
            return []

        n = len(chunks)
        ids = chunk_ids or [f"chunk_{i}" for i in range(n)]
        dids: list[str | None] = doc_ids or [None] * n

        scores = self._predict(query, chunks)
        return [
            self._make_record(score, ids[i], dids[i])
            for i, score in enumerate(scores)
        ]

    # ------------------------------------------------------------------
    # RetrievalRelevanceScorer adapter — plugs into RetrievalEvidenceProfilerV0
    # ------------------------------------------------------------------

    def score(self, query: str, chunk_text: str):  # -> RetrievalRelevanceScore
        """Single-pair scoring — implements RetrievalRelevanceScorer protocol.

        Allows direct injection into RetrievalEvidenceProfilerV0(relevance_scorer=...).
        """
        from raggov.analyzers.retrieval.relevance import RetrievalRelevanceScore
        from raggov.models.retrieval_evidence import QueryRelevanceLabel, RelevanceMethod

        if not self.is_available():
            return RetrievalRelevanceScore(
                score=None,
                label=QueryRelevanceLabel.UNKNOWN,
                method=RelevanceMethod.CROSS_ENCODER.value,
                error="sentence-transformers not installed",
            )
        try:
            scores = self._predict(query, [chunk_text])
            if len(scores) > 1:
                raw = scores[min(self._single_score_cursor, len(scores) - 1)]
                self._single_score_cursor += 1
            else:
                raw = scores[0]
            label = self._score_to_query_label(raw)
            return RetrievalRelevanceScore(
                score=round(raw, 4),
                label=label,
                method=RelevanceMethod.CROSS_ENCODER.value,
                explanation=(
                    f"cross_encoder score={raw:.4f} label={label.value} "
                    f"(relevant>={self._relevant_threshold}, "
                    f"partial>={self._partial_threshold}, "
                    "uncalibrated)"
                ),
            )
        except Exception as exc:
            from raggov.analyzers.retrieval.relevance import RetrievalRelevanceScore
            from raggov.models.retrieval_evidence import QueryRelevanceLabel, RelevanceMethod
            return RetrievalRelevanceScore(
                score=None,
                label=QueryRelevanceLabel.UNKNOWN,
                method=RelevanceMethod.CROSS_ENCODER.value,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(
                    self._model_name,
                    local_files_only=not self._allow_model_downloads,
                )
            except Exception as exc:
                if not self._allow_model_downloads:
                    raise RuntimeError(
                        "cross_encoder model not cached locally. "
                        "Pre-download the model or set allow_model_downloads=True."
                    ) from exc
                raise
        return self._model

    def _predict(self, query: str, texts: list[str]) -> list[float]:
        model = self._load_model()
        pairs = [[query, text] for text in texts]
        raw = model.predict(pairs, batch_size=self._batch_size)
        try:
            return [float(s) for s in raw]
        except TypeError:
            return [float(raw)]

    def _score_to_label(self, score: float) -> str:
        if score >= self._relevant_threshold:
            return "relevant"
        if score >= self._partial_threshold:
            return "partial"
        return "irrelevant"

    def _score_to_query_label(self, score: float):
        from raggov.models.retrieval_evidence import QueryRelevanceLabel
        label_str = self._score_to_label(score)
        return {
            "relevant": QueryRelevanceLabel.RELEVANT,
            "partial": QueryRelevanceLabel.PARTIAL,
            "irrelevant": QueryRelevanceLabel.IRRELEVANT,
        }[label_str]

    def _score_chunks(self, query: str, chunks: list[Any]) -> list[ExternalSignalRecord]:
        texts = [c.text for c in chunks]
        scores = self._predict(query, texts)
        return [
            self._make_record(
                scores[i],
                chunks[i].chunk_id,
                chunks[i].source_doc_id,
            )
            for i in range(len(chunks))
        ]

    def _make_record(
        self, score: float, chunk_id: str, doc_id: str | None
    ) -> ExternalSignalRecord:
        label = self._score_to_label(score)
        return ExternalSignalRecord(
            provider=self.provider,
            signal_type=ExternalSignalType.retrieval_relevance,
            metric_name="cross_encoder_relevance",
            value=round(score, 4),
            label=label,
            explanation=(
                f"cross_encoder score={score:.4f} → label={label!r} "
                f"(relevant_threshold={self._relevant_threshold}, "
                f"partial_threshold={self._partial_threshold}, "
                f"model={self._model_name}, uncalibrated)"
            ),
            affected_chunk_ids=[chunk_id],
            affected_doc_ids=[doc_id] if doc_id else [],
            limitations=list(_BASE_LIMITATIONS) + [
                f"relevant_threshold={self._relevant_threshold} (uncalibrated default)",
                f"partial_threshold={self._partial_threshold} (uncalibrated default)",
                f"model={self._model_name}",
            ],
        )
