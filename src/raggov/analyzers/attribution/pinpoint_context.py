"""Lightweight NCV pinpoint context bridge for A2P.

This module lets A2P consume NCV-derived PinpointFinding objects without
changing A2P scoring or introducing any calibrated/gating semantics.
"""

from __future__ import annotations

import json
from typing import Any

from raggov.models.diagnosis import AnalyzerResult
from raggov.models.pinpoint import PinpointFinding
from raggov.models.run import RAGRun
from raggov.pinpoint.from_ncv import build_pinpoint_findings_from_ncv_report


_NODE_STAGE_MAP = {
    "query_understanding": "retrieval",
    "parser_validity": "parsing",
    "retrieval_coverage": "retrieval",
    "retrieval_precision": "retrieval",
    "context_assembly": "retrieval",
    "version_validity": "retrieval",
    "claim_support": "grounding",
    "citation_support": "grounding",
    "answer_completeness": "generation",
    "security_risk": "security",
}

_NODE_FAILURE_TYPE_MAP = {
    "query_understanding": "SCOPE_VIOLATION",
    "parser_validity": "PARSER_STRUCTURE_LOSS",
    "retrieval_coverage": "INSUFFICIENT_CONTEXT",
    "retrieval_precision": "RETRIEVAL_ANOMALY",
    "context_assembly": "INCONSISTENT_CHUNKS",
    "version_validity": "STALE_RETRIEVAL",
    "claim_support": "UNSUPPORTED_CLAIM",
    "citation_support": "CITATION_MISMATCH",
    "answer_completeness": "GENERATION_IGNORE",
    "security_risk": "SUSPICIOUS_CHUNK",
}


def get_pinpoint_findings_for_a2p(
    run: RAGRun,
    prior_results: list[AnalyzerResult] | None = None,
    config: dict[str, Any] | None = None,
) -> list[PinpointFinding]:
    """Resolve NCV-derived PinpointFinding objects for A2P.

    Resolution order:
    1. `config["pinpoint_findings"]`
    2. `run.metadata["reports"]["ncv_report"]`
    3. `prior_results` / configured prior results carrying `ncv_report`
    """
    config = config or {}

    injected = config.get("pinpoint_findings")
    findings = _coerce_findings(injected)
    if findings:
        return findings

    ncv_report = _ncv_report_from_run(run)
    if ncv_report is None:
        ncv_report = _ncv_report_from_results(prior_results or [])
    if ncv_report is None:
        ncv_report = _ncv_report_from_results(config.get("weighted_prior_results", []))
    if ncv_report is None:
        ncv_report = _ncv_report_from_results(config.get("prior_results", []))
    if ncv_report is None:
        return []

    try:
        return build_pinpoint_findings_from_ncv_report(ncv_report)
    except Exception:
        return []


def summarize_pinpoint_findings_for_a2p(findings: list[PinpointFinding]) -> dict[str, Any]:
    """Return a compact, honesty-preserving A2P pinpoint summary."""
    if not findings:
        return {
            "pinpoint_available": False,
            "recommended_for_gating": False,
            "calibration_status": "uncalibrated",
            "limitations": ["ncv_pinpoint_unavailable"],
        }

    primary = findings[0]
    location = primary.location

    evidence_for = [_compact_evidence(item) for item in primary.evidence_for[:5]]
    evidence_against = [_compact_evidence(item) for item in primary.evidence_against[:5]]
    limitations = list(dict.fromkeys(list(location.limitations) + list(primary.missing_evidence)))

    return {
        "pinpoint_available": True,
        "primary_ncv_node": location.ncv_node,
        "pipeline_stage": location.pipeline_stage or _NODE_STAGE_MAP.get(location.ncv_node or ""),
        "failure_type": location.failure_type or _NODE_FAILURE_TYPE_MAP.get(location.ncv_node or ""),
        "affected_claim_ids": list(location.claim_ids),
        "affected_chunk_ids": list(location.chunk_ids),
        "affected_doc_ids": list(location.doc_ids),
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "missing_evidence": list(primary.missing_evidence),
        "fallback_heuristics_used": list(primary.fallback_heuristics_used),
        "calibration_status": primary.calibration_status,
        "recommended_for_gating": False,
        "limitations": limitations,
    }


def _coerce_findings(raw: Any) -> list[PinpointFinding]:
    if not raw:
        return []
    findings: list[PinpointFinding] = []
    for item in raw:
        try:
            if isinstance(item, PinpointFinding):
                findings.append(item)
            else:
                findings.append(PinpointFinding.model_validate(item))
        except Exception:
            continue
    return findings


def _ncv_report_from_run(run: RAGRun) -> dict[str, Any] | None:
    metadata = getattr(run, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    reports = metadata.get("reports")
    if isinstance(reports, dict):
        report = reports.get("ncv_report")
        if isinstance(report, dict):
            return report
    return None


def _ncv_report_from_results(raw_results: Any) -> dict[str, Any] | None:
    if not isinstance(raw_results, list):
        return None
    for item in raw_results:
        if not isinstance(item, AnalyzerResult):
            continue
        if isinstance(item.ncv_report, dict):
            return item.ncv_report
        if item.analyzer_name != "NCVPipelineVerifier":
            continue
        if not item.evidence:
            continue
        try:
            payload = json.loads(item.evidence[0])
        except Exception:
            continue
        if isinstance(payload, dict) and "node_results" in payload:
            return payload
    return None


def _compact_evidence(item: Any) -> str:
    signal_name = getattr(item, "signal_name", None)
    interpretation = getattr(item, "interpretation", None)
    value = getattr(item, "value", None)
    if signal_name and value is not None:
        return f"{signal_name}={value}"
    if signal_name:
        return str(signal_name)
    if interpretation:
        return str(interpretation)
    return str(item)
