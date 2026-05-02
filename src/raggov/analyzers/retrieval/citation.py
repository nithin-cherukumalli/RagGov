"""Analyzer for citation quality and source traceability."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
from raggov.models.run import RAGRun

_REMEDIATION = (
    "Audit citation logic. Model cited documents outside the retrieved "
    "context window."
)


class CitationMismatchAnalyzer(BaseAnalyzer):
    """Detect citations that reference documents outside retrieved context.

    v0 is a heuristic baseline — not calibrated, not NLI-based, not
    research-faithful (not RAGChecker / RefChecker).  Useful for early
    warning only.  Not recommended for production gating.

    When a RetrievalEvidenceProfile is attached to the run, phantom citations
    are read from profile.phantom_citation_doc_ids (pre-computed by the
    profiler).  Otherwise, the legacy cited_doc_ids vs retrieved_doc_ids
    comparison is used.  The analysis_source field on the returned result
    records which path was taken.
    """

    weight = 1.0

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        profile: RetrievalEvidenceProfile | None = run.retrieval_evidence_profile
        if profile is not None:
            return self._from_profile(profile)
        return self._legacy(run)

    # ------------------------------------------------------------------
    # Profile path
    # ------------------------------------------------------------------

    def _from_profile(self, profile: RetrievalEvidenceProfile) -> AnalyzerResult:
        if not profile.phantom_citation_doc_ids:
            return self._pass(analysis_source="retrieval_evidence_profile")
        evidence = [
            f"[profile] phantom citation: {doc_id}"
            for doc_id in profile.phantom_citation_doc_ids
        ]
        return self._fail(
            FailureType.CITATION_MISMATCH,
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
        if not run.cited_doc_ids:
            return self.skip("no cited_doc_ids provided")

        retrieved_doc_ids = {chunk.source_doc_id for chunk in run.retrieved_chunks}
        phantom_citations = [
            doc_id for doc_id in run.cited_doc_ids if doc_id not in retrieved_doc_ids
        ]

        if phantom_citations:
            return self._fail(
                FailureType.CITATION_MISMATCH,
                FailureStage.RETRIEVAL,
                [f"phantom citation: {doc_id}" for doc_id in phantom_citations],
                _REMEDIATION,
                analysis_source="legacy_heuristic_fallback",
            )

        return self._pass(analysis_source="legacy_heuristic_fallback")
