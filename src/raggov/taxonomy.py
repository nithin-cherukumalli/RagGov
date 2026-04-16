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
    FailureType.PRIVACY_VIOLATION: (
        "The query requests sensitive personal information that should not be disclosed."
    ),
    FailureType.LOW_CONFIDENCE: (
        "Available confidence signals indicate the output may not be trustworthy."
    ),
    FailureType.PARSER_STRUCTURE_LOSS: (
        "Document parser lost structural information before chunking."
    ),
    FailureType.CHUNKING_BOUNDARY_ERROR: (
        "Chunking boundaries split logical content units incorrectly."
    ),
    FailureType.EMBEDDING_DRIFT: (
        "Embedding model collapsed semantically on near-duplicate documents."
    ),
    FailureType.RETRIEVAL_DEPTH_LIMIT: (
        "Top-k limit excluded critical chunks from retrieval results."
    ),
    FailureType.RERANKER_FAILURE: (
        "Reranker demoted the most relevant chunks in final ordering."
    ),
    FailureType.GENERATION_IGNORE: (
        "LLM ignored relevant context and generated from parametric memory."
    ),
    FailureType.CLEAN: "No diagnostic failure was detected.",
}


STAGE_DESCRIPTIONS: dict[FailureStage, str] = {
    FailureStage.PARSING: "Document parsing and structure extraction",
    FailureStage.CHUNKING: "Text chunking and segmentation",
    FailureStage.EMBEDDING: "Embedding generation and vector encoding",
    FailureStage.RETRIEVAL: "Document retrieval from vector store or search index",
    FailureStage.RERANKING: "Reranking of retrieved candidates",
    FailureStage.GROUNDING: "Claim verification against retrieved evidence",
    FailureStage.SUFFICIENCY: "Context adequacy check before generation",
    FailureStage.GENERATION: "Final answer generation from context",
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
    FailureType.PRIVACY_VIOLATION: (
        "Reject the query, abstain from answering, or redact sensitive details from context."
    ),
    FailureType.LOW_CONFIDENCE: (
        "Consider abstaining, re-retrieving, or requesting human review."
    ),
    FailureType.PARSER_STRUCTURE_LOSS: (
        "Use a structure-preserving parser that maintains document hierarchy and tables."
    ),
    FailureType.CHUNKING_BOUNDARY_ERROR: (
        "Adjust chunking strategy to preserve logical units like paragraphs and sections."
    ),
    FailureType.EMBEDDING_DRIFT: (
        "Review embedding model for semantic collapse on near-duplicates or switch models."
    ),
    FailureType.RETRIEVAL_DEPTH_LIMIT: (
        "Increase top-k retrieval limit to include more candidate chunks."
    ),
    FailureType.RERANKER_FAILURE: (
        "Audit reranker behavior or disable reranking if it degrades quality."
    ),
    FailureType.GENERATION_IGNORE: (
        "Strengthen grounding instructions or use a more instruction-following model."
    ),
    FailureType.CLEAN: "No remediation is required.",
}


FAILURE_PRIORITY: list[FailureType] = [
    FailureType.PROMPT_INJECTION,
    FailureType.SUSPICIOUS_CHUNK,
    FailureType.PRIVACY_VIOLATION,
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.PARSER_STRUCTURE_LOSS,
    FailureType.CHUNKING_BOUNDARY_ERROR,
    FailureType.EMBEDDING_DRIFT,
    FailureType.RETRIEVAL_DEPTH_LIMIT,
    FailureType.RERANKER_FAILURE,
    FailureType.GENERATION_IGNORE,
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
        FailureType.PRIVACY_VIOLATION,
    }
