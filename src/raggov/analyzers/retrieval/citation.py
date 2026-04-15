"""Analyzer for citation quality and source traceability."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


class CitationMismatchAnalyzer(BaseAnalyzer):
    """Detect citations that reference documents outside retrieved context."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
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
                "Audit citation logic. Model cited documents outside the retrieved "
                "context window.",
            )

        return self._pass()
