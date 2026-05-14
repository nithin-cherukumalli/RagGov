"""Pinpoint-driven causal-chain builder for A2P.

Builds structured CausalChain objects from existing PinpointFinding and A2P
outputs without changing candidate scoring, calibration, or gating semantics.
"""

from __future__ import annotations

from typing import Any

from raggov.models.pinpoint import (
    CausalChain,
    PinpointFinding,
    PinpointLocation,
    TrustDecision,
)


_DEFAULT_CHAIN_MAP: dict[str, dict[str, str]] = {
    "parser_validity": {
        "causal_hypothesis": "parser_or_ingestion_damage",
        "abduct": "Parser likely damaged structure or provenance before retrieval could use the document correctly.",
        "act": "Repair parser, chunking, and provenance profile handling so structural metadata is preserved.",
        "predict": "Parser repair should reduce downstream retrieval, citation, and grounding failures caused by damaged source structure.",
    },
    "retrieval_coverage": {
        "causal_hypothesis": "retrieval_coverage_gap",
        "abduct": "Needed evidence was not retrieved, or the corpus lacks the required source material.",
        "act": "Improve retrieval recall, query rewriting, corpus coverage, and metadata filtering.",
        "predict": "Unsupported and insufficient-context failures should reduce if the missing evidence is retrieved or added to the corpus.",
    },
    "retrieval_precision": {
        "causal_hypothesis": "retrieval_noise_or_query_context_mismatch",
        "abduct": "Irrelevant or noisy chunks entered the context and diluted useful evidence.",
        "act": "Improve reranking, filtering, retrieval precision, and query routing.",
        "predict": "Context precision and claim support should improve as noisy chunks are filtered out.",
    },
    "context_assembly": {
        "causal_hypothesis": "conflicting_or_badly_assembled_context",
        "abduct": "Retrieved chunks conflict or were assembled in a misleading way before generation.",
        "act": "Add contradiction checks, deduplicate overlapping evidence, and improve context packing.",
        "predict": "Contradiction and context inconsistency failures should reduce after better context assembly.",
    },
    "version_validity": {
        "causal_hypothesis": "stale_or_invalid_source_usage",
        "abduct": "The cited or retrieved source is superseded, expired, withdrawn, deprecated, or not yet effective.",
        "act": "Enforce effective-date and supersession filtering during retrieval and citation selection.",
        "predict": "Stale-source and post-rationalized citation failures should reduce when invalid versions are filtered out.",
    },
    "claim_support": {
        "causal_hypothesis": "generation_grounding_failure",
        "abduct": "The answer claim is unsupported or contradicted despite available context.",
        "act": "Improve grounding constraints, abstention behavior, and claim verification before finalizing the answer.",
        "predict": "Unsupported and contradicted claims should reduce with stricter grounding controls.",
    },
    "citation_support": {
        "causal_hypothesis": "citation_support_failure",
        "abduct": "The claim may be correct or supported elsewhere, but the cited source does not support it.",
        "act": "Require claim-citation alignment and a citation verifier before returning the final answer.",
        "predict": "Citation mismatch and phantom citation failures should reduce after citation alignment is enforced.",
    },
    "answer_completeness": {
        "causal_hypothesis": "answer_completeness_failure",
        "abduct": "The answer omitted a required entity, date, number, or condition despite the question demanding it.",
        "act": "Add answer completeness checks and structured answer requirements before finalization.",
        "predict": "Incomplete answers should reduce when completeness requirements are enforced.",
    },
    "security_risk": {
        "causal_hypothesis": "adversarial_or_unsafe_context",
        "abduct": "Retrieved content contains prompt injection, poisoning, or unsafe instructions.",
        "act": "Quarantine risky chunks and add retrieval-time security screening before generation.",
        "predict": "Security-risk and unsafe-context failures should reduce when adversarial content is filtered out.",
    },
    "query_understanding": {
        "causal_hypothesis": "query_understanding_failure",
        "abduct": "Query intent was misunderstood before retrieval was executed.",
        "act": "Improve intent classification, query rewriting, and routing logic.",
        "predict": "Retrieval and grounding quality should improve when the system better understands the query intent.",
    },
}


def build_causal_chains_from_a2p(
    pinpoint_findings: list[PinpointFinding],
    claim_attributions: list[Any] | None = None,
    claim_attributions_v2: list[Any] | None = None,
    candidate_causes: list[Any] | None = None,
) -> list[CausalChain]:
    """Build structured CausalChain objects from pinpoint + A2P attribution.

    v1 returns at most one causal chain rooted at the first PinpointFinding.
    """
    if not pinpoint_findings:
        return []

    finding = pinpoint_findings[0]
    inferred = infer_causal_hypothesis_from_pinpoint(finding)
    attribution_payload = _select_attribution_payload(
        claim_attributions=claim_attributions,
        claim_attributions_v2=claim_attributions_v2,
        candidate_causes=candidate_causes,
    )

    causal_hypothesis = attribution_payload.get("primary_cause") or inferred["causal_hypothesis"]
    abduct = attribution_payload.get("abduct") or inferred["abduct"]
    act = (
        attribution_payload.get("act")
        or attribution_payload.get("recommended_fix")
        or inferred["act"]
    )
    predict = attribution_payload.get("predict") or inferred["predict"]

    alternative_explanations = list(attribution_payload.get("alternative_explanations", []))
    if not alternative_explanations:
        alternative_explanations = [str(item) for item in finding.missing_evidence[:3]]

    heuristic_score = attribution_payload.get("heuristic_score", finding.heuristic_score)

    return [
        CausalChain(
            chain_id=f"a2p_chain_{finding.location.location_id}",
            root_location=finding.location,
            downstream_locations=list(finding.alternative_locations),
            causal_hypothesis=str(causal_hypothesis),
            abduct=str(abduct),
            act=str(act),
            predict=str(predict),
            evidence_for=list(finding.evidence_for),
            evidence_against=list(finding.evidence_against),
            alternative_explanations=alternative_explanations,
            heuristic_score=float(heuristic_score) if isinstance(heuristic_score, (int, float)) else None,
            calibrated_confidence=None,
            calibration_status="uncalibrated",
        )
    ]


def infer_causal_hypothesis_from_pinpoint(finding: PinpointFinding) -> dict[str, str]:
    """Infer a conservative causal hypothesis from the pinpoint node."""
    node = (finding.location.ncv_node or "").strip().lower()
    default = {
        "causal_hypothesis": "unknown_upstream_failure",
        "abduct": "The pinpointed failure location suggests an upstream pipeline issue, but the cause is not yet specific.",
        "act": "Inspect the pinpointed node and adjacent upstream evidence before changing downstream components.",
        "predict": "Addressing the upstream failure should reduce the observed downstream symptoms.",
    }
    return {**default, **_DEFAULT_CHAIN_MAP.get(node, {})}


def _select_attribution_payload(
    *,
    claim_attributions: list[Any] | None,
    claim_attributions_v2: list[Any] | None,
    candidate_causes: list[Any] | None,
) -> dict[str, Any]:
    if claim_attributions_v2:
        attribution = claim_attributions_v2[0]
        payload: dict[str, Any] = {
            "primary_cause": getattr(attribution, "primary_cause", None),
            "recommended_fix": getattr(attribution, "recommended_fix", None),
            "act": getattr(attribution, "recommended_fix", None),
            "alternative_explanations": list(getattr(attribution, "secondary_causes", []) or []),
        }
        evidence_summary = list(getattr(attribution, "evidence_summary", []) or [])
        if evidence_summary:
            payload["abduct"] = evidence_summary[0]
            if len(evidence_summary) > 1:
                payload["predict"] = evidence_summary[-1]
        chain_candidates = list(getattr(attribution, "candidate_causes", []) or [])
        if chain_candidates:
            primary = chain_candidates[0]
            payload["abduct"] = getattr(primary, "abduct", payload.get("abduct"))
            payload["act"] = getattr(primary, "act", payload.get("act"))
            payload["predict"] = getattr(primary, "predict", payload.get("predict"))
            payload["heuristic_score"] = getattr(primary, "heuristic_score", None)
        return payload

    if claim_attributions:
        attribution = claim_attributions[0]
        return {
            "primary_cause": getattr(attribution, "primary_cause", None),
            "abduct": getattr(attribution, "abduct", None),
            "act": getattr(attribution, "act", None),
            "predict": getattr(attribution, "predict", None),
            "alternative_explanations": list(getattr(attribution, "candidate_causes", []) or [])[1:],
        }

    if candidate_causes:
        candidate = candidate_causes[0]
        return {
            "primary_cause": getattr(candidate, "cause_type", None),
            "abduct": getattr(candidate, "abduct", None),
            "act": getattr(candidate, "act", None),
            "predict": getattr(candidate, "predict", None),
            "heuristic_score": getattr(candidate, "heuristic_score", None),
        }

    return {}


def summarize_causal_chains_for_a2p(chains: list[CausalChain]) -> dict[str, Any]:
    """Return a compact causal-chain summary for A2P evidence output."""
    if not chains:
        return {
            "causal_chain_available": False,
            "calibration_status": "uncalibrated",
            "recommended_for_gating": False,
        }

    chain = chains[0]
    return {
        "causal_chain_available": True,
        "root_node": chain.root_location.ncv_node,
        "root_cause": chain.causal_hypothesis,
        "abduct": chain.abduct,
        "act": chain.act,
        "predict": chain.predict,
        "alternative_explanations": list(chain.alternative_explanations),
        "calibration_status": chain.calibration_status,
        "recommended_for_gating": False,
    }


def build_conservative_trust_decision(
    pinpoint_findings: list[PinpointFinding],
    causal_chains: list[CausalChain],
) -> TrustDecision | None:
    """Return a non-gating trust decision for structured pinpoint outputs."""
    if not pinpoint_findings and not causal_chains:
        return None

    fallback_heuristics_used: list[str] = []
    unmet_requirements = ["calibration_unavailable", "human_review_required"]
    for finding in pinpoint_findings:
        fallback_heuristics_used.extend(finding.fallback_heuristics_used)
        if finding.calibration_status != "uncalibrated":
            unmet_requirements.append("unexpected_calibration_state")

    if any(fallback_heuristics_used):
        decision = "human_review"
        reason = "Structured pinpoint and causal chain outputs are heuristic and include fallback heuristics."
    else:
        decision = "warn"
        reason = "Structured pinpoint and causal chain outputs are available but remain uncalibrated and advisory."

    return TrustDecision(
        decision=decision,
        reason=reason,
        recommended_for_gating=False,
        human_review_required=True,
        blocking_eligible=False,
        unmet_requirements=list(dict.fromkeys(unmet_requirements)),
        calibration_status="uncalibrated",
        fallback_heuristics_used=list(dict.fromkeys(fallback_heuristics_used)),
    )
