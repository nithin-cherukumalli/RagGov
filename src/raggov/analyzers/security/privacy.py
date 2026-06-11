"""Analyzer for private information requests that should result in abstention."""

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

MEDICAL_PRIVACY_PATTERNS = (
    "medical",
    "condition",
    "diagnosis",
    "treatment",
    "migraine",
    "therapy",
    "prescription",
    "medication",
    "disability",
)

PERSON_SUBJECT_RE = re.compile(
    r"\b(?:employee|patient|member|customer|resident|applicant)\s+#?\d+\b",
    re.IGNORECASE,
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

        medical_evidence = self._medical_privacy_evidence(run)

        if not matched_patterns and medical_evidence is None:
            return self._pass()

        evidence = []
        if matched_patterns:
            evidence.append(
                f"Query contains private information pattern(s): {', '.join(matched_patterns)}"
            )
        if medical_evidence is not None:
            evidence.append(medical_evidence)

        return self._fail(
            failure_type=FailureType.PRIVACY_VIOLATION,
            stage=FailureStage.SECURITY,
            evidence=evidence,
            remediation=REMEDIATION,
        )

    def _medical_privacy_evidence(self, run: RAGRun) -> str | None:
        """Detect obvious employee/person medical disclosures in retrieved context or answer."""
        texts = [run.query, run.final_answer]
        texts.extend(chunk.text for chunk in run.retrieved_chunks)
        combined = "\n".join(texts)
        combined_lower = combined.lower()

        if not PERSON_SUBJECT_RE.search(combined):
            return None
        if not any(pattern in combined_lower for pattern in MEDICAL_PRIVACY_PATTERNS):
            return None

        return "Answer or retrieved context contains employee/person medical privacy evidence"
