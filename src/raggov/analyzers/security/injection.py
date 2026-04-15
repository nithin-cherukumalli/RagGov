"""Analyzer for prompt injection and instruction override risks."""

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


PATTERNS = [
    r"ignore (all |previous |above |prior )?(instructions?|prompts?|context|rules?)",
    r"disregard (the |all )?(above|previous|prior|following)",
    r"you are now",
    r"new (persona|role|identity|instructions?)",
    r"act as (a |an )?(different|new)?",
    r"forget (everything|all|what)",
    r"system prompt",
    r"<\|.*\|>",
    r"\[INST\]|\[/INST\]",
    r"###\s*(instruction|system|human|assistant)",
    r"\b(send|forward|email|POST|GET|fetch)\b.{0,30}\b(password|secret|key|token|credential)\b",
    r"reveal (your |the )?(system|prompt|instruction)",
    r"repeat (everything|all|the prompt)",
    r"\b(DAN|jailbreak|unrestricted mode|developer mode)\b",
    r"pretend (there are no|you have no) (rules|restrictions|guidelines)",
]

REMEDIATION = (
    "Retrieved chunk(s) contain instruction-like content consistent with prompt "
    "injection. Sanitize corpus or add a pre-retrieval content filter."
)


class PromptInjectionAnalyzer(BaseAnalyzer):
    """Scan retrieved chunks for prompt injection signals."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        evidence: list[str] = []
        total_hits = 0

        for chunk in run.retrieved_chunks:
            matches = [
                pattern
                for pattern in PATTERNS
                if re.search(pattern, chunk.text, flags=re.IGNORECASE)
            ]
            if matches:
                total_hits += len(matches)
                evidence.append(
                    f"{chunk.chunk_id}: {len(matches)} hit(s): {'; '.join(matches)}"
                )

        if total_hits == 0:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                security_risk=SecurityRisk.NONE,
            )

        risk_threshold = int(self.config.get("risk_threshold", 1))
        warn_threshold = int(self.config.get("warn_threshold", 1))
        if total_hits >= risk_threshold:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
                security_risk=SecurityRisk.HIGH,
                evidence=evidence,
                remediation=REMEDIATION,
            )
        if total_hits >= warn_threshold:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.PROMPT_INJECTION,
                stage=FailureStage.SECURITY,
                security_risk=SecurityRisk.LOW,
                evidence=evidence,
                remediation=REMEDIATION,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            security_risk=SecurityRisk.NONE,
        )
