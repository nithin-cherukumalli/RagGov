"""Attribution trace extraction for A2P v2.

Aggregates structured diagnostic evidence from all prior analyzers.
"""

from __future__ import annotations

import json
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

    # Diagnosis mode metadata
    diagnosis_mode: str | None = None
    external_signals_used: list[str] = field(default_factory=list)

    # Retrieval diagnosis report signals
    has_retrieval_diagnosis_report: bool = False
    retrieval_primary_failure_type: str | None = None

    # NCV report signals
    has_ncv_report: bool = False
    ncv_first_failing_node: str | None = None
    ncv_downstream_failure_chain: list[str] = field(default_factory=list)
    ncv_bottleneck_description: str | None = None

    # External verifier providers (for evidence labelling)
    claim_verifier_provider: str | None = None
    citation_verifier_provider: str | None = None
    retrieval_signal_provider: str | None = None

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

    # Extract diagnosis mode metadata
    diagnosis_mode: str | None = None
    external_signals_used: list[str] = []
    for r in prior_results:
        if hasattr(r, "diagnosis_mode") and r.diagnosis_mode:
            diagnosis_mode = r.diagnosis_mode
            break

    # Extract retrieval diagnosis report signals
    has_retrieval_diagnosis_report = False
    retrieval_primary_failure_type: str | None = None
    retrieval_diag = _retrieval_diagnosis_from_prior(run, prior_results)
    if retrieval_diag is not None:
        has_retrieval_diagnosis_report = True
        retrieval_primary_failure_type = getattr(
            getattr(retrieval_diag, "primary_failure_type", None), "value", None
        ) or str(getattr(retrieval_diag, "primary_failure_type", ""))
        if retrieval_primary_failure_type:
            external_signals_used.append(f"retrieval_diagnosis:{retrieval_primary_failure_type}")
    else:
        trace_notes.append("no_retrieval_diagnosis_report_available")

    # Extract NCV report signals
    has_ncv_report = False
    ncv_first_failing_node: str | None = None
    ncv_downstream_failure_chain: list[str] = []
    ncv_bottleneck_description: str | None = None
    ncv_result = _analyzer_result_by_name(prior_results, "NCVPipelineVerifier")
    if ncv_result is not None and ncv_result.evidence:
        try:
            ncv_report_data = json.loads(ncv_result.evidence[0])
            has_ncv_report = True
            ncv_first_failing_node = ncv_report_data.get("first_failing_node")
            raw_chain = ncv_report_data.get("downstream_failure_chain", [])
            ncv_downstream_failure_chain = [
                (n.get("value", n) if isinstance(n, dict) else str(n)) for n in raw_chain
            ]
            ncv_bottleneck_description = ncv_report_data.get("bottleneck_description")
            if ncv_first_failing_node:
                external_signals_used.append(f"ncv_first_failing:{ncv_first_failing_node}")
        except (json.JSONDecodeError, AttributeError, TypeError):
            trace_notes.append("ncv_report_parse_failed")
    else:
        trace_notes.append("no_ncv_report_available")

    # Extract claim verifier provider from grounding bundle or claim results
    claim_verifier_provider: str | None = None
    grounding_result = _analyzer_result_by_name(prior_results, "ClaimGroundingAnalyzer")
    if grounding_result is not None:
        bundle = grounding_result.grounding_evidence_bundle
        if bundle is not None:
            for rec in getattr(bundle, "claim_evidence_records", []):
                method = getattr(rec, "verifier_method", None)
                if method and method not in {"unknown", "heuristic"}:
                    claim_verifier_provider = method
                    external_signals_used.append(f"claim_verifier:{method}")
                    break
        if claim_verifier_provider is None:
            for cr in claim_results:
                method = getattr(cr, "verification_method", None)
                if method and method not in {"unknown", "heuristic", None}:
                    claim_verifier_provider = method
                    break

    # Extract citation verifier provider from CitationFaithfulnessReport
    citation_verifier_provider: str | None = None
    cit_faith_report = _citation_faithfulness_report_from_prior(run, prior_results)
    if cit_faith_report is not None:
        for rec in getattr(cit_faith_report, "records", []):
            provider = getattr(rec, "external_signal_provider", None)
            if provider:
                citation_verifier_provider = provider
                external_signals_used.append(f"citation_verifier:{provider}")
                break

    # Extract retrieval signal provider from run metadata
    retrieval_signal_provider: str | None = None
    run_meta = getattr(run, "metadata", None) or {}
    for ext_result in run_meta.get("external_evaluation_results", []):
        provider = getattr(ext_result, "provider", None)
        if provider:
            retrieval_signal_provider = str(provider)
            external_signals_used.append(f"retrieval_signal:{retrieval_signal_provider}")
            break

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
        diagnosis_mode=diagnosis_mode,
        external_signals_used=external_signals_used,
        has_retrieval_diagnosis_report=has_retrieval_diagnosis_report,
        retrieval_primary_failure_type=retrieval_primary_failure_type,
        has_ncv_report=has_ncv_report,
        ncv_first_failing_node=ncv_first_failing_node,
        ncv_downstream_failure_chain=ncv_downstream_failure_chain,
        ncv_bottleneck_description=ncv_bottleneck_description,
        claim_verifier_provider=claim_verifier_provider,
        citation_verifier_provider=citation_verifier_provider,
        retrieval_signal_provider=retrieval_signal_provider,
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


def _retrieval_diagnosis_from_prior(run: RAGRun, prior_results: list[AnalyzerResult]) -> Any:
    """Extract RetrievalDiagnosisReport from run or prior results."""
    if getattr(run, "retrieval_diagnosis_report", None) is not None:
        return run.retrieval_diagnosis_report
    for result in prior_results:
        if getattr(result, "retrieval_diagnosis_report", None) is not None:
            return result.retrieval_diagnosis_report
    return None


def _citation_faithfulness_report_from_prior(run: RAGRun, prior_results: list[AnalyzerResult]) -> Any:
    """Extract CitationFaithfulnessReport from run or prior results."""
    if getattr(run, "citation_faithfulness_report", None) is not None:
        return run.citation_faithfulness_report
    for result in prior_results:
        if getattr(result, "citation_faithfulness_report", None) is not None:
            return result.citation_faithfulness_report
    return None
