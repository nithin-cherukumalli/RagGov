"""Primary and secondary cause selection for A2P v2.

Ranks scored candidates and selects primary + secondary causes.
"""

from __future__ import annotations

from raggov.models.diagnosis import CandidateCause, FailureStage


def select_primary_and_secondary_causes(
    scored_candidates: list[CandidateCause],
) -> tuple[str, list[str]]:
    """Select primary cause and secondary causes from scored candidates.

    Primary: highest-ranked candidate
    Secondary: other plausible causes above threshold

    Ranking criteria:
    1. Heuristic score (primary factor)
    2. Stage severity (GENERATION > GROUNDING > RETRIEVAL > SECURITY)
    3. Evidence directness (more evidence_for, fewer evidence_against)
    4. Analyzer support count

    Important: Don't discard stale/citation when contradiction is primary.
    """
    if not scored_candidates:
        return ("unknown", [])

    # Sort by composite ranking
    ranked = sorted(
        scored_candidates,
        key=lambda c: _composite_rank_score(c),
        reverse=True,
    )

    # Primary: top-ranked candidate
    primary = ranked[0].cause_type

    # Secondary: remaining candidates above threshold
    secondary_threshold = 0.30  # Minimum score to be considered plausible
    secondary = [
        c.cause_type
        for c in ranked[1:]
        if (c.heuristic_score or 0.0) >= secondary_threshold
    ]

    # Special case: preserve citation/freshness as secondary when generation is primary
    # This addresses the requirement: "Don't discard stale/citation when contradiction is primary"
    if primary == "generation_contradicted_retrieved_evidence":
        governance_causes = {
            "citation_mismatch",
            "stale_source_usage",
            "post_rationalized_citation",
        }
        governance_candidates = [c for c in ranked if c.cause_type in governance_causes]
        for gov_candidate in governance_candidates:
            if gov_candidate.cause_type not in secondary:
                # Add governance issue even if below threshold
                secondary.append(gov_candidate.cause_type)

    return (primary, secondary)


def _composite_rank_score(candidate: CandidateCause) -> float:
    """Compute composite ranking score for candidate selection.

    Components:
    - Heuristic score (60% weight)
    - Stage severity (20% weight)
    - Evidence balance (10% weight)
    - Analyzer support (10% weight)
    """
    heuristic_score = candidate.heuristic_score or 0.0

    stage_score = _stage_severity_score(candidate.stage)
    evidence_balance = _evidence_balance_score(candidate)
    analyzer_support = _analyzer_support_score(candidate)

    composite = (
        0.60 * heuristic_score
        + 0.20 * stage_score
        + 0.10 * evidence_balance
        + 0.10 * analyzer_support
    )

    return composite


def _stage_severity_score(stage: FailureStage) -> float:
    """Score failure stage by severity (later stages = higher severity)."""
    severity_map = {
        FailureStage.RETRIEVAL: 0.4,
        FailureStage.GROUNDING: 0.6,
        FailureStage.GENERATION: 0.8,
        FailureStage.SECURITY: 0.5,  # Medium severity, but not always root cause
    }
    return severity_map.get(stage, 0.3)


def _evidence_balance_score(candidate: CandidateCause) -> float:
    """Score based on evidence_for vs evidence_against balance.

    More evidence_for = stronger support
    More evidence_against = weaker support
    """
    evidence_for_count = len(candidate.evidence_for)
    evidence_against_count = len(candidate.evidence_against)

    if evidence_for_count == 0:
        return 0.0

    # Balance ratio: for / (for + against)
    balance = evidence_for_count / (evidence_for_count + evidence_against_count)

    # Normalize to [0.0, 1.0] range
    # Strong balance (4 for, 0 against) = 1.0
    # Weak balance (2 for, 2 against) = 0.5
    # Very weak (1 for, 3 against) = 0.25
    return min(1.0, balance)


def _analyzer_support_score(candidate: CandidateCause) -> float:
    """Score based on analyzer support count.

    More supporting analyzers = stronger evidence
    Contradicting analyzers reduce score
    """
    supporting_count = len(candidate.supporting_analyzers)
    contradicting_count = len(candidate.contradicting_analyzers)

    if supporting_count == 0:
        return 0.0

    # Net support score
    net_support = supporting_count - (0.5 * contradicting_count)
    net_support = max(0.0, net_support)

    # Normalize: 3+ supporting analyzers = 1.0
    normalized = min(1.0, net_support / 3.0)

    return normalized


def build_evidence_summary(
    primary_cause: CandidateCause,
    secondary_causes: list[CandidateCause],
) -> list[str]:
    """Build concise evidence summary for attribution output.

    Includes:
    - Primary cause evidence (top 3 items)
    - Secondary causes (name + top evidence item each)
    """
    summary: list[str] = []

    # Primary cause evidence (top 3)
    summary.append(f"PRIMARY: {primary_cause.cause_type}")
    for evidence in primary_cause.evidence_for[:3]:
        summary.append(f"  - {evidence}")

    # Secondary causes (top evidence item each)
    if secondary_causes:
        summary.append("SECONDARY:")
        for sec_cause in secondary_causes[:3]:  # Limit to 3 secondary
            summary.append(f"  - {sec_cause.cause_type}")
            if sec_cause.evidence_for:
                summary.append(f"    {sec_cause.evidence_for[0]}")

    return summary


def build_composite_fix_recommendation(
    primary_cause: CandidateCause,
    secondary_causes: list[CandidateCause],
) -> tuple[str, str]:
    """Build composite fix recommendation addressing primary + secondary causes.

    Returns:
        (recommended_fix, recommended_fix_category)
    """
    # Primary fix action
    primary_fix = primary_cause.act

    # Category based on primary stage
    category_map = {
        FailureStage.RETRIEVAL: "improve_retrieval",
        FailureStage.GROUNDING: "improve_grounding",
        FailureStage.GENERATION: "improve_generation",
        FailureStage.SECURITY: "improve_security",
    }
    category = category_map.get(primary_cause.stage, "unknown")

    # Composite fix includes secondary mitigations
    if secondary_causes:
        secondary_actions = [
            f"Additionally: {sec.act[:80]}..." if len(sec.act) > 80 else f"Additionally: {sec.act}"
            for sec in secondary_causes[:2]  # Limit to 2 secondary actions
        ]
        composite_fix = f"{primary_fix} " + " ".join(secondary_actions)
    else:
        composite_fix = primary_fix

    return (composite_fix, category)
