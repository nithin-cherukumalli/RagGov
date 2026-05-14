"""Analyzer for corpus poisoning and malicious source risks."""

from __future__ import annotations

import re
from typing import Final

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun


ANSWER_STEERING_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)^(the\s+)?(correct\s+|right\s+|only\s+)?(answer|response)\s+is\b",
        "answer_assertion",
    ),
    (
        r"(?i)\b(you\s+should|you\s+must|always)\s+(say|respond|answer|state)\b",
        "response_directive",
    ),
    (r"(?i)\breplace\s+\[.*?\]\s+with\b", "template_fill"),
    (r"(?i)\bsponsored\s+content\b", "sponsored_answer_steering"),
    (r"(?i)<\s*hidden[_-]?payload\s*>", "hidden_payload_marker"),
    (r"(?i)\breset\s*=\s*bypass\b", "bypass_payload"),
]

REMEDIATION = (
    "Chunk exhibits dual-condition poisoning pattern (high retrieval score + "
    "answer-steering content). Quarantine and investigate source document."
)


class PoisoningHeuristicAnalyzer(BaseAnalyzer):
    """Detect dual-condition poisoning signals in retrieved chunks."""

    weight = 0.95

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        scores = [chunk.score for chunk in run.retrieved_chunks if chunk.score is not None]
        if not scores:
            return self.skip("no retrieval scores available")

        dual_condition_evidence: list[str] = []
        steering_only_evidence: list[str] = []

        for chunk in run.retrieved_chunks:
            matches = self._answer_steering_matches(chunk.text)
            if self._has_unlikely_token_distribution(chunk.text):
                matches.append("unlikely_token_distribution")
            if not matches:
                continue
            score = chunk.score or 0.0
            evidence = f"{chunk.chunk_id} score={score:.2f} matched: {'; '.join(matches)}"
            if self._is_score_anomalous(score, scores) or any(
                marker in matches
                for marker in (
                    "sponsored_answer_steering",
                    "hidden_payload_marker",
                    "bypass_payload",
                    "unlikely_token_distribution",
                )
            ):
                dual_condition_evidence.append(evidence)
            else:
                steering_only_evidence.append(evidence)

        if dual_condition_evidence:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.SUSPICIOUS_CHUNK,
                stage=FailureStage.SECURITY,
                security_risk=SecurityRisk.HIGH,
                evidence=dual_condition_evidence,
                remediation=REMEDIATION,
            )
        if steering_only_evidence:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.SUSPICIOUS_CHUNK,
                stage=FailureStage.SECURITY,
                security_risk=SecurityRisk.LOW,
                evidence=steering_only_evidence,
                remediation=REMEDIATION,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            security_risk=SecurityRisk.NONE,
        )

    def _answer_steering_matches(self, text: str) -> list[str]:
        return [
            label
            for pattern, label in ANSWER_STEERING_PATTERNS
            if re.search(pattern, text)
        ]

    def _has_unlikely_token_distribution(self, text: str) -> bool:
        tokens = re.findall(r"[A-Za-z]{5,}", text.lower())
        if len(tokens) < 5:
            return False
        return sum(1 for token in tokens if len(set(token)) <= 2) >= 4

    def _is_score_anomalous(self, chunk_score: float, all_scores: list[float]) -> bool:
        """Return whether a chunk score is an outlier within the retrieved set."""
        if len(all_scores) < 3:
            threshold = float(self.config.get("absolute_score_threshold", 0.92))
            return chunk_score > threshold

        sorted_scores = sorted(all_scores)
        q1 = self._percentile(sorted_scores, 0.25)
        q3 = self._percentile(sorted_scores, 0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        return chunk_score > upper_fence

    def _percentile(self, sorted_values: list[float], quantile: float) -> float:
        """Return a linearly interpolated percentile for a sorted numeric sample."""
        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (len(sorted_values) - 1) * quantile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        fraction = position - lower_index
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        return lower_value + (upper_value - lower_value) * fraction
