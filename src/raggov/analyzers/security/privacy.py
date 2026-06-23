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

# Keywords that indicate a query is asking for sensitive personal information.
# Kept as the public name for backward-compat; it is the union of STRONG + WEAK below.
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

# STRONG patterns are inherently sensitive — they fire regardless of context.
_STRONG_PRIVATE_PATTERNS = (
    "ssn",
    "social security",
    "bank account",
    "credit card",
    "password",
    "secret",
    "confidential",
)
# WEAK patterns are also ordinary topical words ("write an email", "private university",
# "iphone" ⊃ "phone"). They only indicate a privacy request when the query targets a specific
# person's data — otherwise they are CLEAN false positives (Phase C inc3). Matched on word
# boundaries so "iphone" no longer matches "phone".
_WEAK_PRIVATE_PATTERNS = (
    "address",
    "phone",
    "telephone",
    "mobile",
    "email",
    "private",
    "personal",
    "home",
)

# Signals that the query is about a SPECIFIC person's data (so a weak pattern is a real request).
_PERSON_POSSESSIVE_RE = re.compile(r"\b[A-Z][a-z]+'s\b")  # John's, Mary's (orig-case query)
_PERSON_TARGET_RE = re.compile(
    r"\b(?:his|her|their|someone's|somebody's|individual|individuals|person|persons|people|"
    r"employee|employees|patient|customer|member|resident|applicant|colleague|coworker|"
    r"co-worker|neighbou?r|user|client)\b",
    re.IGNORECASE,
)


def _has_person_target(run: RAGRun) -> bool:
    if _PERSON_POSSESSIVE_RE.search(run.query):
        return True
    if _PERSON_TARGET_RE.search(run.query):
        return True
    return bool(PERSON_SUBJECT_RE.search(run.query))


def _word_match(patterns: tuple[str, ...], text: str) -> list[str]:
    return [p for p in patterns if re.search(r"\b" + re.escape(p) + r"\b", text)]

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

        # STRONG patterns are inherently sensitive; WEAK patterns only count as a privacy request
        # when the query targets a specific person (Phase C inc3 — kills CLEAN false positives like
        # "write an email", "private university", "iphone").
        strong = _word_match(_STRONG_PRIVATE_PATTERNS, query_lower)
        weak = _word_match(_WEAK_PRIVATE_PATTERNS, query_lower)
        matched_patterns = list(strong)
        if weak and _has_person_target(run):
            matched_patterns.extend(weak)

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
