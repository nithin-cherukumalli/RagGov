"""Analyzer for anomalous retrieval or response behavior."""

from __future__ import annotations

import math
import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


REMEDIATION = (
    "Retrieval pattern shows statistical anomalies. Investigate for adversarial "
    "document injection or corpus poisoning."
)


class RetrievalAnomalyAnalyzer(BaseAnalyzer):
    """Detect suspicious retrieval score and duplication patterns."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        scored_chunks = [chunk for chunk in run.retrieved_chunks if chunk.score is not None]
        if not scored_chunks:
            return self.skip("no retrieval scores available")

        evidence: list[str] = []
        evidence.extend(self._score_outliers(scored_chunks))
        evidence.extend(self._near_duplicates(run.retrieved_chunks))
        evidence.extend(self._score_cliff(scored_chunks))

        if evidence:
            return self._warn(
                FailureType.RETRIEVAL_ANOMALY,
                FailureStage.SECURITY,
                evidence,
                REMEDIATION,
            )

        return self._pass()

    def _score_outliers(self, chunks: list[RetrievedChunk]) -> list[str]:
        if len(chunks) < 2:
            return []
        threshold = float(self.config.get("zscore_threshold", 2.0))
        scores = [chunk.score for chunk in chunks if chunk.score is not None]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        stddev = math.sqrt(variance)
        if stddev == 0:
            return []

        evidence = []
        for chunk in chunks:
            if chunk.score is None:
                continue
            zscore = (chunk.score - mean) / stddev
            if zscore > threshold:
                evidence.append(
                    f"score outlier {chunk.chunk_id} score={chunk.score:.2f} zscore={zscore:.2f}"
                )
        return evidence

    def _near_duplicates(self, chunks: list[RetrievedChunk]) -> list[str]:
        threshold = float(self.config.get("duplicate_overlap_threshold", 0.85))
        evidence = []
        for index, left in enumerate(chunks):
            left_terms = self._terms(left.text)
            if not left_terms:
                continue
            for right in chunks[index + 1 :]:
                right_terms = self._terms(right.text)
                if not right_terms:
                    continue
                overlap = len(left_terms & right_terms) / min(
                    len(left_terms), len(right_terms)
                )
                if overlap > threshold:
                    evidence.append(
                        f"near duplicate chunks {left.chunk_id} and {right.chunk_id} "
                        f"overlap={overlap:.2f}"
                    )
        return evidence

    def _score_cliff(self, chunks: list[RetrievedChunk]) -> list[str]:
        if len(chunks) < 2:
            return []
        sorted_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.score if chunk.score is not None else float("-inf"),
            reverse=True,
        )
        top = sorted_chunks[0]
        second = sorted_chunks[1]
        if top.score is None or second.score is None:
            return []

        threshold = float(self.config.get("score_cliff_threshold", 0.45))
        if top.score > 0.95 and second.score < 0.5 and top.score - second.score > threshold:
            return [
                f"score cliff between {top.chunk_id} score={top.score:.2f} and "
                f"{second.chunk_id} score={second.score:.2f}"
            ]
        return []

    def _terms(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
