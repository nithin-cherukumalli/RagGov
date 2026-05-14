"""Analyzer for private information requests that should result in abstention."""

from __future__ import annotations

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun

# Keywords that indicate a query is asking for sensitive personal information
PRIVATE_PATTERNS = (
    "address",
    "phone",
    "telephone",
    "mobile",
    "email",
    "private",
    "personal",
    "home",
    "ssn",
    "social security",
    "bank account",
    "credit card",
    "password",
    "secret",
    "confidential",
)

REMEDIATION = (
    "The query requests sensitive personal information. Abstain from answering "
    "or redact sensitive details from retrieved context."
)


class PrivacyAnalyzer(BaseAnalyzer):
    """Detect queries requesting sensitive personal information."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        query_lower = run.query.lower()

        if "password" in query_lower and any(
            token in query_lower for token in ("reset", "lost", "forgot", "2fa")
        ):
            return self._pass()

        # Check if query contains private patterns
        matched_patterns = [
            pattern for pattern in PRIVATE_PATTERNS if pattern in query_lower
        ]

        if not matched_patterns:
            return self._pass()

        evidence = [
            f"Query contains private information pattern(s): {', '.join(matched_patterns)}"
        ]

        return self._fail(
            failure_type=FailureType.PRIVACY_VIOLATION,
            stage=FailureStage.SECURITY,
            evidence=evidence,
            remediation=REMEDIATION,
        )
