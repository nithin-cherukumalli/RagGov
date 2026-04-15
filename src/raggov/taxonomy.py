"""Taxonomy definitions for RagGov diagnoses and analyzer categories."""

from __future__ import annotations

from raggov.models.diagnosis import FailureStage, FailureType


FAILURE_MESSAGES: dict[FailureType, str] = {
    FailureType.STALE_RETRIEVAL: (
        "Retrieved documents are outdated and may not reflect current information."
    ),
    FailureType.SCOPE_VIOLATION: (
        "Retrieved documents appear off-topic for the user's query."
    ),
    FailureType.CITATION_MISMATCH: (
        "The answer cites sources that were not present in the retrieved context."
    ),
    FailureType.INCONSISTENT_CHUNKS: (
        "Retrieved chunks contain potentially inconsistent or conflicting information."
    ),
    FailureType.INSUFFICIENT_CONTEXT: (
        "Retrieved context does not contain enough information to answer reliably."
    ),
    FailureType.UNSUPPORTED_CLAIM: (
        "The answer contains claims that are not supported by retrieved evidence."
    ),
    FailureType.CONTRADICTED_CLAIM: (
        "The answer contains claims contradicted by retrieved evidence."
    ),
    FailureType.PROMPT_INJECTION: (
        "Retrieved content contains instruction-like text consistent with prompt injection."
    ),
    FailureType.SUSPICIOUS_CHUNK: (
        "A retrieved chunk shows signs of answer-steering or corpus poisoning."
    ),
    FailureType.RETRIEVAL_ANOMALY: (
        "Retrieval results show statistical anomalies that may indicate manipulation."
    ),
    FailureType.LOW_CONFIDENCE: (
        "Available confidence signals indicate the output may not be trustworthy."
    ),
    FailureType.CLEAN: "No diagnostic failure was detected.",
}


STAGE_DESCRIPTIONS: dict[FailureStage, str] = {
    FailureStage.RETRIEVAL: "Document retrieval from vector store or search index",
    FailureStage.GROUNDING: "Claim verification against retrieved evidence",
    FailureStage.SUFFICIENCY: "Context adequacy check before generation",
    FailureStage.SECURITY: "Retrieved content safety and integrity check",
    FailureStage.CONFIDENCE: "Output confidence and uncertainty estimation",
    FailureStage.UNKNOWN: "Unclassified or unknown pipeline stage",
}


DEFAULT_REMEDIATIONS: dict[FailureType, str] = {
    FailureType.STALE_RETRIEVAL: (
        "Re-index stale documents or add freshness filtering to retrieval."
    ),
    FailureType.SCOPE_VIOLATION: (
        "Review query preprocessing, embedding model behavior, and index quality."
    ),
    FailureType.CITATION_MISMATCH: (
        "Audit citation generation and restrict citations to retrieved documents."
    ),
    FailureType.INCONSISTENT_CHUNKS: (
        "Review retrieved chunks for contradictions and consider deduplication or reranking."
    ),
    FailureType.INSUFFICIENT_CONTEXT: (
        "Expand retrieval, broaden the query, or abstain from answering."
    ),
    FailureType.UNSUPPORTED_CLAIM: (
        "Add source verification or remove claims not supported by retrieved evidence."
    ),
    FailureType.CONTRADICTED_CLAIM: (
        "Remove contradicted claims and re-check grounding against retrieved evidence."
    ),
    FailureType.PROMPT_INJECTION: (
        "Sanitize retrieved content or add prompt-injection filtering before generation."
    ),
    FailureType.SUSPICIOUS_CHUNK: (
        "Quarantine suspicious documents and investigate their source provenance."
    ),
    FailureType.RETRIEVAL_ANOMALY: (
        "Investigate retrieval anomalies for adversarial injection or corpus poisoning."
    ),
    FailureType.LOW_CONFIDENCE: (
        "Consider abstaining, re-retrieving, or requesting human review."
    ),
    FailureType.CLEAN: "No remediation is required.",
}


FAILURE_PRIORITY: list[FailureType] = [
    FailureType.PROMPT_INJECTION,
    FailureType.SUSPICIOUS_CHUNK,
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.CONTRADICTED_CLAIM,
    FailureType.INSUFFICIENT_CONTEXT,
    FailureType.UNSUPPORTED_CLAIM,
    FailureType.CITATION_MISMATCH,
    FailureType.STALE_RETRIEVAL,
    FailureType.SCOPE_VIOLATION,
    FailureType.INCONSISTENT_CHUNKS,
    FailureType.LOW_CONFIDENCE,
    FailureType.CLEAN,
]


def should_have_answered(failure_type: FailureType) -> bool:
    """Return whether the system should have answered despite the failure type."""
    return failure_type not in {
        FailureType.PROMPT_INJECTION,
        FailureType.SUSPICIOUS_CHUNK,
        FailureType.INSUFFICIENT_CONTEXT,
    }
