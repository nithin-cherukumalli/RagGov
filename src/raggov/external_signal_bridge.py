"""Bridge converting external evaluator signals into targeted native diagnostic probes."""

from __future__ import annotations

import uuid
from typing import Any

from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.external_diagnosis import ExternalSignalDiagnosisProbe
from raggov.models.run import RAGRun
from raggov.evaluators.base import ExternalEvaluationResult


_UNKNOWN_RELEVANCE_LABELS = {"", "none", "null", "unknown"}


def _get_chunk_relevance_label(chunk_profile: Any) -> str | None:
    """Return the best available chunk relevance label without assuming model shape."""
    for field_name in ("query_relevance_label", "native_relevance_label", "relevance_label"):
        value = _get_profile_field(chunk_profile, field_name)
        label = _normalize_label(value)
        if label is not None:
            return label
    return None


def _get_profile_field(chunk_profile: Any, field_name: str) -> Any:
    if isinstance(chunk_profile, dict):
        return chunk_profile.get(field_name)
    try:
        return getattr(chunk_profile, field_name)
    except AttributeError:
        return None


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    raw = value.value if hasattr(value, "value") else value
    label = str(raw).lower()
    if label in _UNKNOWN_RELEVANCE_LABELS:
        return None
    return label


def _chunk_profile_id(chunk_profile: Any, field_name: str) -> str | None:
    value = _get_profile_field(chunk_profile, field_name)
    if value is None:
        return None
    return str(value)


def _get_claim_record_label(record: Any) -> str | None:
    for field_name in ("claim_label", "verification_label", "label"):
        label = _normalize_label(_get_profile_field(record, field_name))
        if label == "insufficient":
            return "unsupported"
        if label is not None:
            return label
    return None


def _has_unsupported_or_contradicted_claims(grounding_bundle: Any) -> bool:
    records = _get_profile_field(grounding_bundle, "claim_evidence_records") or []
    return any(
        _get_claim_record_label(record) in {"unsupported", "contradicted"}
        for record in records
    )


def build_external_signal_diagnosis_probes(
    run: RAGRun,
    analyzer_results: list[AnalyzerResult],
) -> list[ExternalSignalDiagnosisProbe]:
    """Inspect external evaluation results and map them to targeted native diagnosis probes."""
    probes: list[ExternalSignalDiagnosisProbe] = []

    eval_results_raw = run.metadata.get("external_evaluation_results", [])
    signals = []
    for item in eval_results_raw:
        if isinstance(item, dict):
            try:
                res = ExternalEvaluationResult.model_validate(item)
                signals.extend(res.signals)
            except Exception:
                continue
        elif isinstance(item, ExternalEvaluationResult):
            signals.extend(item.signals)

    if not signals:
        return probes

    # Gather native context once for all probes
    native_relevance_labels = []
    any_relevance_label_available = False
    noisy_suspected = False
    phantom_citations = False
    if run.retrieval_evidence_profile:
        for chunk in run.retrieval_evidence_profile.chunks:
            relevance_label = _get_chunk_relevance_label(chunk)
            if relevance_label is None:
                continue
            any_relevance_label_available = True
            if relevance_label == "irrelevant":
                chunk_id = _chunk_profile_id(chunk, "chunk_id")
                if chunk_id is not None:
                    native_relevance_labels.append(chunk_id)
        cross_encoder_metrics = _get_profile_field(
            run.retrieval_evidence_profile,
            "cross_encoder_metrics",
        )
        if isinstance(cross_encoder_metrics, dict):
            if cross_encoder_metrics.get("mean_relevance") is not None:
                if cross_encoder_metrics["mean_relevance"] < 0.5:
                    noisy_suspected = True
                    
    retrieval_diag = None
    citation_faithfulness = None
    grounding_bundle = None
    ncv_report = None
    
    for r in analyzer_results:
        if r.analyzer_name == "RetrievalDiagnosisAnalyzerV0":
            retrieval_diag = r.retrieval_diagnosis_report
            if retrieval_diag and hasattr(retrieval_diag, "primary_failure_type") and retrieval_diag.primary_failure_type.value == "retrieval_noise":
                noisy_suspected = True
        elif r.analyzer_name == "CitationFaithfulnessAnalyzerV0":
            citation_faithfulness = r.citation_faithfulness_report
            if citation_faithfulness and citation_faithfulness.phantom_citation_doc_ids:
                phantom_citations = True
        elif r.analyzer_name == "ClaimGroundingAnalyzer":
            grounding_bundle = r.grounding_evidence_bundle
        elif r.analyzer_name == "NCVPipelineVerifier":
            ncv_report = r.ncv_report

    for signal in signals:
        provider = signal.provider.value if hasattr(signal.provider, "value") else signal.provider
        metric_name = signal.metric_name
        val = signal.value
        signal_type_val = signal.signal_type.value if hasattr(signal.signal_type, "value") else signal.signal_type
        provenance_fields = {
            "affected_claim_ids": list(signal.affected_claim_ids),
            "affected_chunk_ids": list(signal.affected_chunk_ids),
            "affected_doc_ids": list(signal.affected_doc_ids),
        }
        
        is_low_score = False
        is_high_score = False
        is_bad_label = False
        
        if isinstance(val, (int, float)):
            if val < 0.5:
                is_low_score = True
            elif val > 0.5:
                is_high_score = True
                
        if isinstance(val, str) and val.lower() in ("unsupported", "contradicted", "does_not_support", "irrelevant"):
            is_bad_label = True

        # A. retrieval_context_precision low
        if (
            metric_name in ("context_precision", "retrieval_context_precision")
            and provider in ("ragas", "ragchecker")
            and is_low_score
        ):
            native_found = []
            native_missing = []
            if native_relevance_labels:
                native_found.append(
                    f"{len(native_relevance_labels)} chunks have relevance label=irrelevant natively "
                    f"(chunk_ids: {', '.join(native_relevance_labels)})"
                )
            else:
                if any_relevance_label_available:
                    native_missing.append("No native chunk relevance label indicated irrelevant context")
                else:
                    native_missing.append("No chunk-level relevance label available")
                
            if noisy_suspected:
                native_found.append("retrieval_diagnosis.noisy_context_suspected=True")
            
            if ncv_report and ncv_report.get("first_failing_node") == "retrieval_precision":
                native_found.append("NCV retrieval_precision failed")
                
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="medium",
                    suspected_pipeline_node="retrieval_precision",
                    suspected_failure_stage=FailureStage.RETRIEVAL.value,
                    suspected_failure_type=FailureType.RETRIEVAL_ANOMALY.value,
                    native_analyzers_to_check=["RetrievalDiagnosisAnalyzerV0", "RetrievalEvidenceProfilerV0", "ScopeViolationAnalyzer", "NCVPipelineVerifier"],
                    native_evidence_found=native_found,
                    native_evidence_missing=native_missing,
                    explanation=f"{provider} {metric_name} is low, suggesting retrieval precision issues. Investigating chunk relevance and noise.",
                    recommended_recheck="Inspect ranking and context filtering.",
                    recommended_fix_category="improve_retrieval_precision",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # B. retrieval_context_recall / claim_recall low
        elif (
            metric_name in ("context_recall", "retrieval_context_recall", "claim_recall", "contextual_recall")
            and provider in ("ragas", "ragchecker", "deepeval")
            and is_low_score
        ):
            native_found = []
            native_missing = []
            if ncv_report and ncv_report.get("first_failing_node") == "retrieval_coverage":
                native_found.append("NCV retrieval_coverage failed")
            
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="high",
                    suspected_pipeline_node="retrieval_coverage",
                    suspected_failure_stage=FailureStage.RETRIEVAL.value,
                    suspected_failure_type=FailureType.INSUFFICIENT_CONTEXT.value,
                    native_analyzers_to_check=["SufficiencyAnalyzer", "ClaimAwareSufficiencyAnalyzer", "RetrievalDiagnosisAnalyzerV0"],
                    native_evidence_found=native_found,
                    native_evidence_missing=native_missing,
                    explanation=f"{provider} {metric_name} is low, suggesting missing context to answer the query fully.",
                    recommended_recheck="Check for missing evidence requirements.",
                    recommended_fix_category="improve_retrieval_recall",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # C. contextual_relevancy low
        elif metric_name == "contextual_relevancy" and provider == "deepeval" and is_low_score:
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="medium",
                    suspected_pipeline_node="retrieval_precision",
                    suspected_failure_stage=FailureStage.RETRIEVAL.value,
                    suspected_failure_type=FailureType.SCOPE_VIOLATION.value,
                    native_analyzers_to_check=["ScopeViolationAnalyzer", "RetrievalEvidenceProfilerV0"],
                    native_evidence_found=[],
                    native_evidence_missing=[],
                    explanation=f"{provider} {metric_name} is low, indicating potential scope violation or retrieval noise.",
                    recommended_recheck="Check if retrieved chunks are within the query scope.",
                    recommended_fix_category="improve_retrieval_precision",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # D. contextual_precision low
        elif metric_name == "contextual_precision" and provider == "deepeval" and is_low_score:
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="medium",
                    suspected_pipeline_node="retrieval_precision",
                    suspected_failure_stage=FailureStage.RERANKING.value,
                    suspected_failure_type=FailureType.RERANKER_FAILURE.value,
                    native_analyzers_to_check=["RetrievalDiagnosisAnalyzerV0", "RetrievalEvidenceProfilerV0"],
                    native_evidence_found=[],
                    native_evidence_missing=["No reranker scores available"],
                    explanation=f"{provider} {metric_name} is low, pointing to potential reranker failure or suboptimal ordering.",
                    recommended_recheck="Inspect the ordering of retrieved chunks.",
                    recommended_fix_category="improve_reranking",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # E. faithfulness low
        elif metric_name == "faithfulness" and provider in ("ragas", "deepeval", "ragchecker") and is_low_score:
            native_found = []
            if grounding_bundle and _has_unsupported_or_contradicted_claims(grounding_bundle):
                native_found.append("Found unsupported or contradicted claims in native grounding bundle.")
            if citation_faithfulness and (
                citation_faithfulness.unsupported_claim_ids or citation_faithfulness.contradicted_claim_ids
            ):
                native_found.append("Citation faithfulness report found unsupported/contradicted claims.")
                
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="high",
                    suspected_pipeline_node="claim_support",
                    suspected_failure_stage=FailureStage.GROUNDING.value,
                    suspected_failure_type=FailureType.UNSUPPORTED_CLAIM.value,
                    native_analyzers_to_check=["ClaimGroundingAnalyzer", "CitationFaithfulnessAnalyzerV0", "NCVPipelineVerifier"],
                    native_evidence_found=native_found,
                    native_evidence_missing=[],
                    explanation=f"{provider} {metric_name} is low, indicating generation drift, unsupported claims, or citation mismatch.",
                    recommended_recheck="Inspect claim-level grounding and citation faithfulness.",
                    recommended_fix_category="improve_generation_grounding",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # F. hallucination high / RefChecker claim contradicted
        elif (
            (metric_name == "hallucination" and provider == "ragchecker" and is_high_score) or
            (metric_name == "claim_support" and provider == "refchecker" and is_bad_label)
        ):
            native_found = []
            if grounding_bundle and _has_unsupported_or_contradicted_claims(grounding_bundle):
                native_found.append("Found unsupported or contradicted claims in native grounding bundle.")

            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="high",
                    suspected_pipeline_node="claim_support",
                    suspected_failure_stage=FailureStage.GROUNDING.value,
                    suspected_failure_type=FailureType.CONTRADICTED_CLAIM.value if is_bad_label and getattr(val, "lower", lambda: "")() == "contradicted" else FailureType.UNSUPPORTED_CLAIM.value,
                    native_analyzers_to_check=["ClaimGroundingAnalyzer", "SemanticEntropyAnalyzer"],
                    native_evidence_found=native_found,
                    native_evidence_missing=[],
                    explanation=f"{provider} {metric_name} indicates hallucination or unsupported claims in the generated response.",
                    recommended_recheck="Investigate generation hallucination and claim support.",
                    recommended_fix_category="improve_generation_grounding",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # G. citation_support does_not_support / contradicted
        elif metric_name in ("citation_support", "citation_missing") and is_bad_label:
            native_found = []
            if citation_faithfulness and (
                citation_faithfulness.unsupported_claim_ids or citation_faithfulness.missing_citation_claim_ids
            ):
                native_found.append("Citation faithfulness report found citation coverage issues.")
            if phantom_citations:
                native_found.append("Phantom citations detected natively.")

            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="high",
                    suspected_pipeline_node="citation_support",
                    suspected_failure_stage=FailureStage.GROUNDING.value,
                    suspected_failure_type=FailureType.CITATION_MISMATCH.value,
                    native_analyzers_to_check=["CitationFaithfulnessAnalyzerV0", "CitationFaithfulnessProbe"],
                    native_evidence_found=native_found,
                    native_evidence_missing=[],
                    explanation=f"{provider} {metric_name} indicates that citations do not support the claims or are missing.",
                    recommended_recheck="Check for citation mismatch or post-rationalized citations.",
                    recommended_fix_category="improve_citation_accuracy",
                    should_block_clean=True,
                    should_trigger_human_review=True,
                )
            )

        # H. cross_encoder irrelevant chunk signals
        elif provider == "cross_encoder" and is_bad_label:
            native_found = []
            if noisy_suspected:
                native_found.append("Retrieval noise natively suspected.")
            probes.append(
                ExternalSignalDiagnosisProbe(
                    probe_id=f"probe_{uuid.uuid4().hex[:8]}",
                    provider=provider,
                    metric_name=metric_name,
                    signal_type=signal_type_val,
                    external_value=val,
                    external_label=signal.label,
                    **provenance_fields,
                    severity="medium",
                    suspected_pipeline_node="retrieval_precision",
                    suspected_failure_stage=FailureStage.RETRIEVAL.value,
                    suspected_failure_type=FailureType.RETRIEVAL_ANOMALY.value,
                    native_analyzers_to_check=["RetrievalDiagnosisAnalyzerV0"],
                    native_evidence_found=native_found,
                    native_evidence_missing=[],
                    explanation=f"Cross-encoder scored a chunk as irrelevant.",
                    recommended_recheck="Inspect chunk-level retrieval noise.",
                    recommended_fix_category="improve_retrieval_precision",
                    should_block_clean=False,  # chunk-level relevance might not block clean if answer is okay
                    should_trigger_human_review=False,
                )
            )

    return probes
