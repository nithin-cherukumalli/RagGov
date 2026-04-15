"""Base analyzer interfaces and shared analyzer types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun


class BaseAnalyzer(ABC):
    """Minimal contract shared by all RagGov analyzers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze a RAG run and return the analyzer result."""

    def name(self) -> str:
        """Return the analyzer name."""
        return self.__class__.__name__

    def skip(self, reason: str) -> AnalyzerResult:
        """Return a skipped analyzer result with the skip reason as evidence."""
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="skip",
            evidence=[reason],
        )

    def _pass(self, evidence: list[str] | None = None) -> AnalyzerResult:
        """Return a passing analyzer result."""
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence or [],
        )

    def _warn(
        self,
        failure_type: FailureType,
        stage: FailureStage,
        evidence: list[str],
        remediation: str,
    ) -> AnalyzerResult:
        """Return a warning analyzer result."""
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="warn",
            failure_type=failure_type,
            stage=stage,
            evidence=evidence,
            remediation=remediation,
        )

    def _fail(
        self,
        failure_type: FailureType,
        stage: FailureStage,
        evidence: list[str],
        remediation: str,
    ) -> AnalyzerResult:
        """Return a failing analyzer result."""
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=failure_type,
            stage=stage,
            evidence=evidence,
            remediation=remediation,
        )
