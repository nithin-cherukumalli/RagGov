"""Analyzer for stale or outdated retrieved context."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import QueryRelevanceLabel, RetrievalEvidenceProfile
from raggov.models.run import RAGRun


# Standard chunk-metadata keys exposed by common RAG frameworks (LangChain,
# LlamaIndex, Haystack) for document freshness. Lower-cased for matching.
_TEMPORAL_METADATA_KEYS: tuple[str, ...] = (
    "effective_date",
    "valid_until",
    "valid_from",
    "as_of",
    "published_at",
    "updated_at",
    "date",
)

_DATE_PREFIX_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _parse_date_prefix(value: object) -> date | None:
    """Parse ``YYYY``, ``YYYY-MM``, or ``YYYY-MM-DD`` prefixes off a value.

    Returns ``None`` on anything unparseable. Pipeline-agnostic — no
    timezone math, no domain assumption.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    match = _DATE_PREFIX_RE.match(value.strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _chunk_effective_date(chunk: RetrievedChunk) -> tuple[date | None, str | None]:
    """Return the chunk's effective date and which metadata key it came from.

    Preference order matches semantic strength: ``effective_date`` >
    ``valid_from`` > ``as_of`` > ``published_at`` > ``updated_at`` >
    ``date`` > ``valid_until``. ``valid_until`` is last because it marks
    end-of-validity rather than the version's own date.
    """
    metadata = chunk.metadata or {}
    # Match keys case-insensitively without mutating the dict.
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    preference = (
        "effective_date",
        "valid_from",
        "as_of",
        "published_at",
        "updated_at",
        "date",
        "valid_until",
    )
    for key in preference:
        if key in lowered:
            parsed = _parse_date_prefix(lowered[key])
            if parsed is not None:
                return parsed, key
    return None, None


def _answer_alignment_score(answer: str, chunk_text: str) -> float:
    """Token-overlap Jaccard-like score in [0, 1].

    Cheap, deterministic, framework-free signal of how textually similar
    the answer is to a chunk. We do not need NLI here — the relative
    ranking across chunks is what matters, not the absolute value.
    """
    answer_tokens = {t.lower() for t in _TOKEN_RE.findall(answer)}
    chunk_tokens = {t.lower() for t in _TOKEN_RE.findall(chunk_text)}
    if not answer_tokens or not chunk_tokens:
        return 0.0
    overlap = answer_tokens & chunk_tokens
    return len(overlap) / len(answer_tokens)


class StaleRetrievalAnalyzer(BaseAnalyzer):
    """Detect retrieved chunks backed by stale corpus entries.

    v0 is a heuristic baseline — not calibrated, not NLI-based, not
    research-faithful (not RAGChecker / RefChecker).  Useful for early
    warning only.  Not recommended for production gating.

    When a RetrievalEvidenceProfile is attached to the run, stale document
    IDs are read from profile.stale_doc_ids (pre-computed by the profiler).
    Otherwise, the legacy age-threshold check against corpus entry timestamps
    is used.  The analysis_source field on the returned result records which
    path was taken.
    """

    weight = 0.95

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        profile: RetrievalEvidenceProfile | None = run.retrieval_evidence_profile
        if profile is not None and profile.stale_doc_ids:
            return self._from_profile(run, profile)

        # The chunk-metadata path is checked next regardless of whether a
        # profile is attached: a profile with no pre-computed stale_doc_ids
        # does not preclude chunk-level temporal evidence of staleness.
        relative = self._from_chunk_metadata(run)
        if relative is not None:
            return relative

        if profile is not None:
            return self._from_profile(run, profile)
        return self._legacy(run)

    # ------------------------------------------------------------------
    # Relative-recency path (chunk-metadata)
    # ------------------------------------------------------------------

    def _from_chunk_metadata(self, run: RAGRun) -> AnalyzerResult | None:
        """Detect staleness from chunk-level temporal metadata.

        Fires when the answer textually aligns with an older retrieved
        chunk while a strictly newer chunk was also retrieved. Returns
        ``None`` if the run lacks the structured signals needed for a
        relative-recency claim, so the caller can fall through to the
        legacy corpus-age path.
        """
        if not run.retrieved_chunks:
            return None

        dated: list[tuple[RetrievedChunk, date, str]] = []
        for chunk in run.retrieved_chunks:
            chunk_date, key = _chunk_effective_date(chunk)
            if chunk_date is not None and key is not None:
                dated.append((chunk, chunk_date, key))

        if len(dated) < 2:
            return None

        # Find the answer-aligned chunk among dated chunks.
        answer = run.final_answer or ""
        scored = [
            (chunk, chunk_date, key, _answer_alignment_score(answer, chunk.text))
            for chunk, chunk_date, key in dated
        ]
        scored.sort(key=lambda t: t[3], reverse=True)
        aligned_chunk, aligned_date, aligned_key, aligned_score = scored[0]

        if aligned_score <= 0.0:
            # No textual alignment with any dated chunk — staleness cannot
            # be asserted relative to the answer.
            return None

        # Find a strictly newer dated chunk.
        newer_candidates = [
            (chunk, chunk_date, key)
            for chunk, chunk_date, key in dated
            if chunk_date > aligned_date
        ]
        if not newer_candidates:
            return None

        min_staleness_days = int(self.config.get("min_staleness_days", 30))
        newer_chunk, newer_date, newer_key = max(
            newer_candidates, key=lambda t: t[1]
        )
        delta_days = (newer_date - aligned_date).days
        if delta_days < min_staleness_days:
            return None

        evidence = [
            (
                f"answer-aligned chunk {aligned_chunk.chunk_id} "
                f"({aligned_key}={aligned_date.isoformat()}) is {delta_days} days "
                f"older than retrieved chunk {newer_chunk.chunk_id} "
                f"({newer_key}={newer_date.isoformat()})"
            ),
            f"answer overlap with stale chunk: {aligned_score:.2f} token coverage",
        ]
        remediation = (
            "Re-rank or filter retrieval by recency: the generator selected a "
            "stale chunk even though a newer chunk was retrieved. Verify the "
            f"chunk metadata key '{aligned_key}' is honored by the retriever or "
            "downstream ranker."
        )
        return self._fail(
            FailureType.STALE_RETRIEVAL,
            FailureStage.RETRIEVAL,
            evidence,
            remediation,
            analysis_source="legacy_heuristic_fallback",
        )

    # ------------------------------------------------------------------
    # Profile path
    # ------------------------------------------------------------------

    def _from_profile(
        self, run: RAGRun, profile: RetrievalEvidenceProfile
    ) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        if not profile.stale_doc_ids:
            return self._pass(analysis_source="retrieval_evidence_profile")

        # Task 14: an age-stale doc is only diagnosis-bearing when it is query-relevant
        # AND a strictly-newer dated alternative was retrieved. An irrelevant stale
        # distractor (not query-relevant) or the freshest retrieved doc itself (no newer
        # alternative) must not promote STALE_RETRIEVAL to primary. Genuine stale cases
        # (a relevant outdated version coexisting with a newer one) are preserved.
        effective_stale = [
            doc_id
            for doc_id in profile.stale_doc_ids
            if self._stale_doc_is_diagnostic(doc_id, run, profile)
        ]
        if not effective_stale:
            return self._pass(analysis_source="retrieval_evidence_profile")

        max_age_days = int(self.config.get("max_age_days", 180))
        evidence = [
            f"[profile] stale document: {doc_id}"
            for doc_id in effective_stale
        ]
        return self._fail(
            FailureType.STALE_RETRIEVAL,
            FailureStage.RETRIEVAL,
            evidence,
            f"Re-index documents older than {max_age_days} days or add freshness "
            "filtering to retrieval.",
            analysis_source="retrieval_evidence_profile",
        )

    def _stale_doc_is_diagnostic(
        self, doc_id: str, run: RAGRun, profile: RetrievalEvidenceProfile
    ) -> bool:
        """A profile-stale doc counts only if query-relevant AND superseded by a newer one."""
        return self._doc_query_relevant(doc_id, profile) and self._has_newer_dated_alternative(
            doc_id, run
        )

    @staticmethod
    def _doc_query_relevant(doc_id: str, profile: RetrievalEvidenceProfile) -> bool:
        labels = [
            cp.query_relevance_label
            for cp in (profile.chunks or [])
            if cp.source_doc_id == doc_id
        ]
        if not labels:
            return True  # no relevance info: do not suppress (preserve legacy behavior)
        return any(
            label in (QueryRelevanceLabel.RELEVANT, QueryRelevanceLabel.PARTIAL)
            for label in labels
        )

    @staticmethod
    def _has_newer_dated_alternative(doc_id: str, run: RAGRun) -> bool:
        dates = {}
        for entry in run.corpus_entries or []:
            ts = getattr(entry, "timestamp", None)
            if ts is not None:
                dates[entry.doc_id] = ts
        own = dates.get(doc_id)
        if own is None:
            return True  # no date info: do not suppress (preserve legacy behavior)
        return any(other != doc_id and ts > own for other, ts in dates.items())

    # ------------------------------------------------------------------
    # Legacy fallback (original v0 logic — preserved exactly)
    # ------------------------------------------------------------------

    def _legacy(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")
        if not run.corpus_entries:
            return self.skip("no corpus metadata available")

        max_age_days = int(self.config.get("max_age_days", 180))
        stale_before = datetime.now(UTC) - timedelta(days=max_age_days)
        corpus_by_doc_id = {entry.doc_id: entry for entry in run.corpus_entries}
        evidence: list[str] = []

        for chunk in run.retrieved_chunks:
            entry = corpus_by_doc_id.get(chunk.source_doc_id)
            if entry is None or entry.timestamp is None:
                continue

            timestamp = entry.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            if timestamp < stale_before:
                age_days = (datetime.now(UTC) - timestamp).days
                evidence.append(f"{entry.doc_id} is {age_days} days old")

        if evidence:
            return self._fail(
                FailureType.STALE_RETRIEVAL,
                FailureStage.RETRIEVAL,
                evidence,
                "Re-index documents older than "
                f"{max_age_days} days or add freshness filtering to retrieval.",
                analysis_source="legacy_heuristic_fallback",
            )

        return self._pass(analysis_source="legacy_heuristic_fallback")
