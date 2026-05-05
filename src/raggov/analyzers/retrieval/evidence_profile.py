"""
RetrievalEvidenceProfilerV0: heuristic-baseline evidence profile builder.

Produces a RetrievalEvidenceProfile from a RAGRun using only the v0 retrieval
heuristics already present in the retrieval analyzer suite.

IMPORTANT:
- v0 is heuristic_baseline and uncalibrated.
- Do not use recommended_for_gating outputs in production gating flows.
- All signals are approximate; see the limitations list on each returned profile.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from raggov.analyzers.retrieval.contradiction import (
    ContradictionDetector,
    ContradictionLabel,
    NegationHeuristicContradictionDetector,
)
from raggov.analyzers.retrieval.freshness import (
    AgeBasedFreshnessEvaluator,
    FreshnessEvaluator,
)
from raggov.analyzers.retrieval.relevance import (
    LexicalOverlapRelevanceScorer,
    RetrievalRelevanceScorer,
)
from raggov.evaluators.base import ExternalSignalRecord, ExternalSignalType
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import (
    CalibrationStatus,
    ChunkEvidenceProfile,
    CitationStatus,
    EvidenceRole,
    FreshnessStatus,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalEvidenceProfile,
    RetrievalMethodType,
)
from raggov.models.run import RAGRun

_EXTERNAL_RELEVANCE_LABEL_MAP = {
    "relevant": QueryRelevanceLabel.RELEVANT,
    "partial": QueryRelevanceLabel.PARTIAL,
    "irrelevant": QueryRelevanceLabel.IRRELEVANT,
}


# Defaults match ScopeViolationAnalyzer and StaleRetrievalAnalyzer exactly.
_DEFAULT_MIN_OVERLAP_RATIO: float = 0.1
_DEFAULT_MAX_AGE_DAYS: int = 180

_LIMITATIONS = [
    "v0 uses lexical overlap, not semantic relevance",
    "v0 contradiction detection uses negation heuristics, not NLI",
    "v0 freshness uses age threshold, not legal/version validity",
    "v0 citation detection checks provenance IDs only, not claim-level support",
]


class RetrievalEvidenceProfilerV0:
    """
    Build a RetrievalEvidenceProfile from a RAGRun using v0 heuristics.

    Signals are drawn directly from existing retrieval analyzers:
    - Query relevance  : lexical overlap (ScopeViolationAnalyzer logic)
    - Phantom citations: cited doc IDs absent from retrieved set (CitationMismatchAnalyzer)
    - Staleness        : age-threshold on corpus entry timestamp (StaleRetrievalAnalyzer)
    - Contradictions   : negation-pair heuristic (InconsistentChunksAnalyzer)

    v0: heuristic_baseline, uncalibrated, NOT recommended for gating.

    Scorer, evaluator, and detector can be overridden via constructor injection
    to plug in v1+ implementations (embeddings, NLI, etc.) without modifying
    this class.  The default implementations preserve v0 heuristic behaviour
    exactly.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        relevance_scorer: Optional[RetrievalRelevanceScorer] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        freshness_evaluator: Optional[FreshnessEvaluator] = None,
    ) -> None:
        cfg = config or {}
        min_overlap = float(cfg.get("min_overlap_ratio", _DEFAULT_MIN_OVERLAP_RATIO))
        max_age = int(cfg.get("max_age_days", _DEFAULT_MAX_AGE_DAYS))

        self._relevance_scorer: RetrievalRelevanceScorer = (
            relevance_scorer or LexicalOverlapRelevanceScorer(min_overlap_ratio=min_overlap)
        )
        self._contradiction_detector: ContradictionDetector = (
            contradiction_detector or NegationHeuristicContradictionDetector()
        )
        self._freshness_evaluator: FreshnessEvaluator = (
            freshness_evaluator or AgeBasedFreshnessEvaluator(max_age_days=max_age)
        )

    def build(self, run: RAGRun) -> RetrievalEvidenceProfile:
        """Produce a RetrievalEvidenceProfile for the given RAGRun."""
        if not run.retrieved_chunks:
            return RetrievalEvidenceProfile(
                run_id=run.run_id,
                method_type=RetrievalMethodType.HEURISTIC_BASELINE,
                calibration_status=CalibrationStatus.UNCALIBRATED,
                recommended_for_gating=False,
                limitations=["no retrieved chunks available"],
            )

        cited_set = set(run.cited_doc_ids)
        corpus_by_doc_id = {e.doc_id: e for e in run.corpus_entries}
        now = datetime.now(UTC)

        chunk_profiles = [
            self._profile_chunk(run.query, c, cited_set, corpus_by_doc_id, now)
            for c in run.retrieved_chunks
        ]

        retrieved_doc_ids = {c.source_doc_id for c in run.retrieved_chunks}
        phantom_citation_doc_ids = [
            d for d in run.cited_doc_ids if d not in retrieved_doc_ids
        ]
        stale_doc_ids = list(
            dict.fromkeys(
                cp.source_doc_id
                for cp in chunk_profiles
                if cp.freshness_status == FreshnessStatus.STALE_BY_AGE
                and cp.source_doc_id
            )
        )
        noisy_chunk_ids = [
            cp.chunk_id
            for cp in chunk_profiles
            if cp.query_relevance_label
            in (QueryRelevanceLabel.IRRELEVANT, QueryRelevanceLabel.PARTIAL)
        ]
        contradictory_pairs = self._contradiction_candidates(run.retrieved_chunks)

        has_issues = any(
            [phantom_citation_doc_ids, stale_doc_ids, noisy_chunk_ids, contradictory_pairs]
        )

        return RetrievalEvidenceProfile(
            run_id=run.run_id,
            overall_retrieval_status="degraded" if has_issues else "ok",
            chunks=chunk_profiles,
            phantom_citation_doc_ids=phantom_citation_doc_ids,
            stale_doc_ids=stale_doc_ids,
            noisy_chunk_ids=noisy_chunk_ids,
            contradictory_pairs=contradictory_pairs,
            method_type=RetrievalMethodType.HEURISTIC_BASELINE,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            recommended_for_gating=False,
            limitations=list(_LIMITATIONS),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_chunk(
        self,
        query: str,
        chunk: RetrievedChunk,
        cited_set: set[str],
        corpus_by_doc_id: dict[str, CorpusEntry],
        now: datetime,
    ) -> ChunkEvidenceProfile:
        rel = self._relevance_scorer.score(query, chunk.text)
        freshness = self._freshness_evaluator.evaluate(
            chunk, corpus_by_doc_id.get(chunk.source_doc_id), now
        )
        return ChunkEvidenceProfile(
            chunk_id=chunk.chunk_id,
            source_doc_id=chunk.source_doc_id,
            query_relevance_label=rel.label,
            query_relevance_score=rel.score,
            native_relevance_label=rel.label,
            native_relevance_score=rel.score,
            relevance_method=RelevanceMethod(rel.method),
            citation_status=self._citation_status(chunk, cited_set),
            freshness_status=freshness.status,
        )

    def _citation_status(
        self, chunk: RetrievedChunk, cited_set: set[str]
    ) -> CitationStatus:
        if not cited_set:
            return CitationStatus.UNKNOWN
        return CitationStatus.CITED if chunk.source_doc_id in cited_set else CitationStatus.UNCITED

    def apply_external_relevance_signals(
        self,
        profile: RetrievalEvidenceProfile,
        signals: list[ExternalSignalRecord],
    ) -> RetrievalEvidenceProfile:
        """Apply external retrieval relevance signals to an existing profile.

        Updates chunk profiles matching signal.affected_chunk_ids with
        external relevance scores and recomputes noisy_chunk_ids.

        Does NOT change lexical-overlap-labeled chunks not mentioned in signals.
        Returns the same profile object (mutated in place) for chaining.

        Signals with signal_type != retrieval_relevance are ignored.
        Missing dependency or empty signals: profile is returned unchanged.
        """
        relevance_signals: dict[str, ExternalSignalRecord] = {}
        for signal in signals:
            if signal.signal_type != ExternalSignalType.retrieval_relevance:
                continue
            for cid in signal.affected_chunk_ids:
                relevance_signals[cid] = signal

        if not relevance_signals:
            return profile

        existing_signal_keys = {
            (
                str(signal.get("provider")),
                str(signal.get("metric_name")),
                tuple(signal.get("affected_chunk_ids", [])),
            )
            for signal in profile.external_signals
            if isinstance(signal, dict)
        }
        for signal in relevance_signals.values():
            payload = signal.model_dump(mode="json")
            key = (
                str(payload.get("provider")),
                str(payload.get("metric_name")),
                tuple(payload.get("affected_chunk_ids", [])),
            )
            if key not in existing_signal_keys:
                profile.external_signals.append(payload)
                existing_signal_keys.add(key)

        updated_chunks: list[ChunkEvidenceProfile] = []
        for cp in profile.chunks:
            sig = relevance_signals.get(cp.chunk_id)
            if sig is None:
                updated_chunks.append(cp)
                continue
            label = _EXTERNAL_RELEVANCE_LABEL_MAP.get(
                str(sig.label or "").lower(), QueryRelevanceLabel.UNKNOWN
            )
            score = float(sig.value) if isinstance(sig.value, (int, float)) else None
            
            # Map relevance to evidence role
            role = cp.evidence_role
            if label == QueryRelevanceLabel.IRRELEVANT:
                role = EvidenceRole.NOISE
            elif label == QueryRelevanceLabel.RELEVANT:
                role = EvidenceRole.NECESSARY_SUPPORT
            elif label == QueryRelevanceLabel.PARTIAL:
                role = EvidenceRole.PARTIAL_SUPPORT

            updated_chunks.append(
                ChunkEvidenceProfile(
                    **{
                        **cp.model_dump(),
                        "query_relevance_label": label,
                        "query_relevance_score": score,
                        "relevance_method": RelevanceMethod.CROSS_ENCODER,
                        "evidence_role": role,
                        "external_provider": sig.provider.value if hasattr(sig.provider, "value") else str(sig.provider),
                        "external_metric_name": sig.metric_name,
                    }
                )
            )

        profile.chunks = updated_chunks
        profile.noisy_chunk_ids = [
            cp.chunk_id
            for cp in profile.chunks
            if cp.query_relevance_label
            in (QueryRelevanceLabel.IRRELEVANT, QueryRelevanceLabel.PARTIAL)
        ]
        has_issues = any(
            [
                profile.phantom_citation_doc_ids,
                profile.stale_doc_ids,
                profile.noisy_chunk_ids,
                profile.contradictory_pairs,
            ]
        )
        profile.overall_retrieval_status = "degraded" if has_issues else "ok"
        profile.calibration_status = CalibrationStatus.UNCALIBRATED_LOCALLY
        profile.method_type = RetrievalMethodType.PRACTICAL_APPROXIMATION
        
        if "external_relevance_signals_applied" not in " ".join(profile.limitations):
            profile.limitations = list(profile.limitations) + [
                "external_relevance_signals_applied: cross-encoder scores are uncalibrated locally"
            ]
        return profile

    def _contradiction_candidates(
        self, chunks: list[RetrievedChunk]
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for i, left in enumerate(chunks):
            for right in chunks[i + 1 :]:
                result = self._contradiction_detector.compare_chunks(left, right)
                if result.label == ContradictionLabel.CONTRADICTION:
                    pairs.append((left.chunk_id, right.chunk_id))
        return pairs
