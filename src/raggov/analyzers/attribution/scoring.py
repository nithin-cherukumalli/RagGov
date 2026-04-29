"""Transparent heuristic scoring for candidate causes.

Scores are explicitly marked as uncalibrated and include basis explanations.
"""

from __future__ import annotations

from raggov.models.diagnosis import CandidateCause


def score_candidate(candidate: CandidateCause) -> CandidateCause:
    """Compute transparent heuristic score for a candidate cause.

    Scoring rules:
    - +0.35 for strong direct analyzer failure
    - +0.25 for claim label match
    - +0.20 for supporting signals (sufficiency/citation/freshness)
    - +0.10 for affected chunk overlap
    - +0.10 for evidence_reason match
    - -0.20 for evidence_against items

    Score capped at [0.0, 1.0].
    """
    score = 0.0
    score_components: list[str] = []

    # Analyzer support
    if candidate.supporting_analyzers:
        analyzer_score = min(0.35, len(candidate.supporting_analyzers) * 0.15)
        score += analyzer_score
        score_components.append(
            f"+{analyzer_score:.2f} from {len(candidate.supporting_analyzers)} supporting analyzer(s)"
        )

    # Affected claims (shows this candidate is relevant)
    if candidate.affected_claims:
        score += 0.25
        score_components.append("+0.25 from affected claim match")

    # Evidence for
    evidence_for_score = min(0.20, len(candidate.evidence_for) * 0.05)
    if evidence_for_score > 0:
        score += evidence_for_score
        score_components.append(
            f"+{evidence_for_score:.2f} from {len(candidate.evidence_for)} evidence_for item(s)"
        )

    # Affected chunks (shows impact scope)
    if candidate.affected_chunk_ids:
        chunk_score = min(0.10, len(candidate.affected_chunk_ids) * 0.03)
        score += chunk_score
        score_components.append(f"+{chunk_score:.2f} from {len(candidate.affected_chunk_ids)} affected chunk(s)")

    # Contradicting analyzers (penalty)
    if candidate.contradicting_analyzers:
        penalty = min(0.20, len(candidate.contradicting_analyzers) * 0.10)
        score -= penalty
        score_components.append(
            f"-{penalty:.2f} from {len(candidate.contradicting_analyzers)} contradicting analyzer(s)"
        )

    # Evidence against (penalty)
    if candidate.evidence_against:
        penalty = min(0.20, len(candidate.evidence_against) * 0.07)
        score -= penalty
        score_components.append(
            f"-{penalty:.2f} from {len(candidate.evidence_against)} evidence_against item(s)"
        )

    # Cap score
    score = max(0.0, min(1.0, score))

    # Build score basis
    score_basis = "; ".join(score_components) if score_components else "no scoring components"
    score_basis = f"Heuristic score {score:.2f}: {score_basis}"

    # Return updated candidate (frozen=False allows modification)
    candidate.heuristic_score = score
    candidate.score_basis = score_basis

    return candidate


def score_all_candidates(candidates: list[CandidateCause]) -> list[CandidateCause]:
    """Score all candidates and return sorted by score descending."""
    scored = [score_candidate(candidate) for candidate in candidates]
    return sorted(scored, key=lambda c: c.heuristic_score or 0.0, reverse=True)
