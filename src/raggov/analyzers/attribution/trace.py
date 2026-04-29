"""Attribution trace extraction for A2P v2.

Aggregates structured diagnostic evidence from all prior analyzers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureType,
    SufficiencyResult,
)
from raggov.models.run import RAGRun


SECURITY_FAILURE_TYPES = {
    FailureType.PROMPT_INJECTION,
    FailureType.SUSPICIOUS_CHUNK,
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.PRIVACY_VIOLATION,
}


@dataclass(frozen=True)
class AttributionTrace:
    """Structured trace of all available diagnostic evidence for A2P v2.

    This trace aggregates typed signals from all prior analyzers to support
    multi-hypothesis counterfactual attribution.
    """

    # Claim signals from ClaimGroundingAnalyzer
    claim_results: list[ClaimResult] = field(default_factory=list)

    # Sufficiency signals
    sufficiency_result: SufficiencyResult | None = None
    sufficiency_method: str | None = None

    # Citation signals
    has_citation_mismatch: bool = False
    phantom_citations: list[str] = field(default_factory=list)
    has_post_rationalized_citation: bool = False
    citation_probe_results: list[dict[str, Any]] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)

    # Freshness signals
    has_stale_retrieval: bool = False
    stale_doc_ids: list[str] = field(default_factory=list)
    stale_evidence: list[str] = field(default_factory=list)

    # Retrieval signals
    chunk_count: int = 0
    avg_score: float = 0.0
    top_score: float = 0.0
    low_score_count: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)

    # Security signals
    has_security_failure: bool = False
    security_failure_types: list[FailureType] = field(default_factory=list)
    security_evidence: list[str] = field(default_factory=list)

    # Failure context
    failure_types_present: set[FailureType] = field(default_factory=set)

    # Trace notes (for missing/unavailable data)
    trace_notes: list[str] = field(default_factory=list)


def extract_attribution_trace(run: RAGRun, prior_results: list[AnalyzerResult]) -> AttributionTrace:
    """Extract structured trace of all diagnostic evidence for A2P v2."""
    trace_notes: list[str] = []

    # Extract claim results
    claim_results = _claim_results_from_prior(prior_results)
    if not claim_results:
        trace_notes.append("no_claim_results_available")

    # Extract sufficiency
    sufficiency_result = _sufficiency_from_prior(prior_results)
    sufficiency_method = sufficiency_result.method if sufficiency_result else None
    if sufficiency_result is None:
        trace_notes.append("no_sufficiency_result_available")

    # Extract citation signals
    citation_mismatch_result = _analyzer_result_by_name(prior_results, "CitationMismatchAnalyzer")
    has_citation_mismatch = citation_mismatch_result is not None and citation_mismatch_result.status == "fail"
    phantom_citations: list[str] = []
    if has_citation_mismatch and citation_mismatch_result:
        phantom_citations = [
            ev.replace("phantom citation: ", "")
            for ev in citation_mismatch_result.evidence
            if "phantom citation:" in ev
        ]

    citation_faith_result = _analyzer_result_by_name(prior_results, "CitationFaithfulnessProbe")
    has_post_rationalized_citation = (
        citation_faith_result is not None and citation_faith_result.status in {"fail", "warn"}
    )
    citation_probe_results = (
        citation_faith_result.citation_probe_results
        if citation_faith_result and citation_faith_result.citation_probe_results
        else []
    )
    cited_doc_ids = run.cited_doc_ids if run.cited_doc_ids else []
    if not cited_doc_ids:
        trace_notes.append("no_citation_ids_available")

    # Extract freshness signals
    stale_result = _analyzer_result_by_name(prior_results, "StaleRetrievalAnalyzer")
    has_stale_retrieval = stale_result is not None and stale_result.status == "fail"
    stale_doc_ids: list[str] = []
    stale_evidence: list[str] = []
    if has_stale_retrieval and stale_result:
        stale_evidence = list(stale_result.evidence)
        # Extract doc IDs from evidence like "doc7 is 2000 days old"
        for ev in stale_result.evidence:
            parts = ev.split()
            if parts and " is " in ev and "days old" in ev:
                stale_doc_ids.append(parts[0])

    if not run.corpus_entries:
        trace_notes.append("no_corpus_timestamp_metadata")

    # Extract retrieval signals
    scores = [c.score for c in run.retrieved_chunks if c.score is not None]
    chunk_count = len(run.retrieved_chunks)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    low_score_count = sum(1 for s in scores if s < 0.6)
    retrieved_doc_ids = list({c.source_doc_id for c in run.retrieved_chunks})

    # Extract security signals
    security_failures = [
        r for r in prior_results if r.status in {"fail", "warn"} and r.failure_type in SECURITY_FAILURE_TYPES
    ]
    has_security_failure = len(security_failures) > 0
    security_failure_types = [r.failure_type for r in security_failures if r.failure_type]
    security_evidence = []
    for r in security_failures:
        security_evidence.extend(r.evidence)

    # Extract failure context
    failure_types_present = {
        r.failure_type for r in prior_results if r.status in {"fail", "warn"} and r.failure_type is not None
    }

    return AttributionTrace(
        claim_results=claim_results,
        sufficiency_result=sufficiency_result,
        sufficiency_method=sufficiency_method,
        has_citation_mismatch=has_citation_mismatch,
        phantom_citations=phantom_citations,
        has_post_rationalized_citation=has_post_rationalized_citation,
        citation_probe_results=citation_probe_results,
        cited_doc_ids=cited_doc_ids,
        has_stale_retrieval=has_stale_retrieval,
        stale_doc_ids=stale_doc_ids,
        stale_evidence=stale_evidence,
        chunk_count=chunk_count,
        avg_score=avg_score,
        top_score=top_score,
        low_score_count=low_score_count,
        retrieved_doc_ids=retrieved_doc_ids,
        has_security_failure=has_security_failure,
        security_failure_types=security_failure_types,
        security_evidence=security_evidence,
        failure_types_present=failure_types_present,
        trace_notes=trace_notes,
    )


def _claim_results_from_prior(prior_results: list[AnalyzerResult]) -> list[ClaimResult]:
    """Extract claim results from ClaimGroundingAnalyzer."""
    for result in prior_results:
        if result.analyzer_name == "ClaimGroundingAnalyzer" and result.claim_results:
            return result.claim_results
    return []


def _sufficiency_from_prior(prior_results: list[AnalyzerResult]) -> SufficiencyResult | None:
    """Extract sufficiency result, preferring ClaimAwareSufficiencyAnalyzer."""
    for result in prior_results:
        if (
            result.analyzer_name == "ClaimAwareSufficiencyAnalyzer"
            and result.sufficiency_result is not None
        ):
            return result.sufficiency_result
    for result in prior_results:
        if result.analyzer_name == "SufficiencyAnalyzer" and result.sufficiency_result is not None:
            return result.sufficiency_result
    return None


def _analyzer_result_by_name(
    prior_results: list[AnalyzerResult], analyzer_name: str
) -> AnalyzerResult | None:
    """Find first analyzer result by name."""
    for result in prior_results:
        if result.analyzer_name == analyzer_name:
            return result
    return None
