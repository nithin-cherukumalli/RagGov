"""Analyzer for corpus poisoning and malicious source risks."""

from __future__ import annotations

import re

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun


ANSWER_STEERING_PATTERNS = [
    r"the answer is",
    r"the correct answer",
    r"you should (say|respond|answer)",
    r"\$[\d,]+|\d+%|\d+ (million|billion|thousand)",
    r"(official|definitive|correct|true) (answer|response|statement)",
]

REMEDIATION = (
    "Chunk exhibits dual-condition poisoning pattern (high retrieval score + "
    "answer-steering content). Quarantine and investigate source document."
)


class PoisoningHeuristicAnalyzer(BaseAnalyzer):
    """Detect simple high-score answer-steering poisoning signals."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        if not any(chunk.score is not None for chunk in run.retrieved_chunks):
            return self.skip("no retrieval scores available")

        dual_condition_evidence: list[str] = []
        steering_only_evidence: list[str] = []

        for chunk in run.retrieved_chunks:
            matches = [
                pattern
                for pattern in ANSWER_STEERING_PATTERNS
                if re.search(pattern, chunk.text, flags=re.IGNORECASE)
            ]
            if not matches:
                continue
            score = chunk.score or 0.0
            evidence = (
                f"{chunk.chunk_id} score={score:.2f} matched: {'; '.join(matches)}"
            )
            if score > 0.85:
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
