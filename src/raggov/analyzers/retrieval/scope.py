"""Analyzer for retrieval scope and relevance boundaries."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import QueryRelevanceLabel, RetrievalEvidenceProfile
from raggov.models.run import RAGRun


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
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
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_REMEDIATION = (
    "Retrieval returned off-topic documents. Review embedding model, "
    "query preprocessing, or index quality."
)


class ScopeViolationAnalyzer(BaseAnalyzer):
    """Detect retrieved chunks that are off-topic relative to the query.

    v0 is a heuristic baseline — not calibrated, not NLI-based, not
    research-faithful (not RAGChecker / RefChecker).  Useful for early
    warning only.  Not recommended for production gating.

    When a RetrievalEvidenceProfile is attached to the run, query relevance
    labels are read from profile.chunks (pre-computed by the profiler).
    Chunks labelled IRRELEVANT trigger warn/fail.  Otherwise, the legacy
    lexical-overlap heuristic is used.  The analysis_source field on the
    returned result records which path was taken.
    """

    weight = 0.75

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

        missing_query_entity = self._missing_quoted_query_entity(run)
        if missing_query_entity:
            return self._fail(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                [f"quoted query entity missing from retrieved context: {missing_query_entity}"],
                _REMEDIATION,
                analysis_source="retrieval_evidence_profile",
            )

        chunk_labels = {cp.chunk_id: cp.query_relevance_label for cp in profile.chunks}
        irrelevant = [
            chunk
            for chunk in run.retrieved_chunks
            if chunk_labels.get(chunk.chunk_id) == QueryRelevanceLabel.IRRELEVANT
        ]

        if not irrelevant:
            return self._pass(analysis_source="retrieval_evidence_profile")

        evidence = [
            f"[profile] {chunk.chunk_id} label=irrelevant" for chunk in irrelevant
        ]

        if len(irrelevant) == len(run.retrieved_chunks):
            return self._fail(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                evidence,
                _REMEDIATION,
                analysis_source="retrieval_evidence_profile",
            )
        return self._warn(
            FailureType.SCOPE_VIOLATION,
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

        query_terms = self._terms(run.query)
        if not query_terms:
            return self.skip("no meaningful query terms available")

        threshold = float(self.config.get("min_overlap_ratio", 0.1))
        low_overlap: list[str] = []
        missing_query_entity = self._missing_quoted_query_entity(run)

        for chunk in run.retrieved_chunks:
            chunk_terms = self._terms(chunk.text)
            overlap_ratio = len(query_terms & chunk_terms) / len(query_terms)
            if overlap_ratio < threshold:
                low_overlap.append(f"{chunk.chunk_id} overlap={overlap_ratio:.2f}")

        if missing_query_entity:
            return self._fail(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                [f"quoted query entity missing from retrieved context: {missing_query_entity}"],
                _REMEDIATION,
                analysis_source="legacy_heuristic_fallback",
            )

        if len(low_overlap) == len(run.retrieved_chunks):
            return self._fail(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                low_overlap,
                _REMEDIATION,
                analysis_source="legacy_heuristic_fallback",
            )
        if low_overlap:
            return self._warn(
                FailureType.SCOPE_VIOLATION,
                FailureStage.RETRIEVAL,
                low_overlap,
                _REMEDIATION,
                analysis_source="legacy_heuristic_fallback",
            )

        return self._pass(analysis_source="legacy_heuristic_fallback")

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS
        }

    def _missing_quoted_query_entity(self, run: RAGRun) -> str | None:
        context = " ".join(chunk.text for chunk in run.retrieved_chunks).lower()
        for match in re.finditer(r"['\"]([^'\"]{3,80})['\"]", run.query):
            entity = match.group(1).strip().lower()
            if entity and entity not in context:
                return entity
        return None
