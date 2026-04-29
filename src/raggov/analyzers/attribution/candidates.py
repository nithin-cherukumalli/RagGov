"""Candidate cause generation for A2P v2.

Generates multiple competing hypotheses for why a claim failed or is risky.
"""

from __future__ import annotations

from raggov.models.diagnosis import CandidateCause, ClaimResult, FailureStage
from raggov.analyzers.attribution.trace import AttributionTrace


def identify_claims_needing_attribution(trace: AttributionTrace) -> list[tuple[ClaimResult, str]]:
    """Identify claims that need attribution (failed or risky).

    Returns list of (claim, reason) tuples where reason explains why attribution is needed.
    """
    claims_needing_attribution: list[tuple[ClaimResult, str]] = []

    for claim in trace.claim_results:
        # Failed claims always need attribution
        if claim.label == "unsupported":
            claims_needing_attribution.append((claim, "claim_unsupported"))
        elif claim.label == "contradicted":
            claims_needing_attribution.append((claim, "claim_contradicted"))
        # Risky claims (entailed but governance issues)
        elif claim.label == "entailed":
            if trace.has_security_failure:
                claims_needing_attribution.append((claim, "security_risk_despite_entailment"))
            elif trace.has_citation_mismatch:
                claims_needing_attribution.append((claim, "citation_invalid_despite_entailment"))
            elif trace.has_post_rationalized_citation:
                claims_needing_attribution.append((claim, "post_rationalized_citation_risk"))
            elif trace.has_stale_retrieval:
                claims_needing_attribution.append((claim, "stale_source_despite_entailment"))
            elif claim.fallback_used:
                claims_needing_attribution.append((claim, "verification_uncertainty"))

    return claims_needing_attribution


def generate_candidate_causes(
    claim: ClaimResult, reason: str, trace: AttributionTrace
) -> list[CandidateCause]:
    """Generate all applicable candidate causes for a failed/risky claim."""
    candidates: list[CandidateCause] = []

    # Generate each candidate type
    candidate = candidate_insufficient_context(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_weak_evidence(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_generation_contradicted(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_stale_source(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_citation_mismatch(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_post_rationalized_citation(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_verification_uncertainty(claim, trace)
    if candidate:
        candidates.append(candidate)

    candidate = candidate_security_adversarial(claim, trace)
    if candidate:
        candidates.append(candidate)

    return candidates


def candidate_insufficient_context(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate insufficient_context_or_retrieval_miss candidate if applicable."""
    if claim.label != "unsupported":
        return None

    if not claim.supporting_chunk_ids and (
        trace.sufficiency_result is not None and not trace.sufficiency_result.sufficient
    ):
        evidence_for = [
            "Claim is unsupported with no supporting chunks",
            f"Sufficiency analyzer reports insufficient context (method: {trace.sufficiency_method})",
        ]

        if trace.sufficiency_result.missing_evidence:
            evidence_for.append(
                f"Missing evidence gaps: {len(trace.sufficiency_result.missing_evidence)} items"
            )

        if trace.avg_score < 0.6:
            evidence_for.append(f"Low average retrieval score: {trace.avg_score:.2f}")

        if trace.chunk_count < 3:
            evidence_for.append(f"Few retrieved chunks: {trace.chunk_count}")

        evidence_against = []
        if claim.candidate_chunk_ids:
            evidence_against.append(
                f"Candidate chunks exist ({len(claim.candidate_chunk_ids)}), "
                "suggesting evidence is present but weak"
            )

        return CandidateCause(
            cause_id=f"{claim.claim_text[:50]}_insufficient_context",
            cause_type="insufficient_context_or_retrieval_miss",
            stage=FailureStage.RETRIEVAL,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            affected_claims=[claim.claim_text],
            affected_chunk_ids=list(claim.candidate_chunk_ids),
            supporting_analyzers=["SufficiencyAnalyzer", "ClaimAwareSufficiencyAnalyzer"],
            contradicting_analyzers=[],
            abduct=(
                "Claim is unsupported with no supporting chunks while sufficiency indicates missing evidence; "
                "the likely hidden cause is retrieval/context insufficiency or a corpus gap."
            ),
            act="Increase retrieval depth (top-k), expand query with synonyms, or add corpus coverage.",
            predict=(
                "If missing evidence exists in corpus and is retrieved, claim should become verifiable. "
                "If evidence does not exist, system should abstain."
            ),
            predicted_fix_effect="would_likely_fix",
            calibration_status="uncalibrated",
        )

    return None


def candidate_weak_evidence(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate weak_or_ambiguous_evidence candidate if applicable."""
    if claim.label != "unsupported":
        return None

    if claim.candidate_chunk_ids and not claim.supporting_chunk_ids:
        evidence_reason_lower = (claim.evidence_reason or "").lower()
        if any(
            term in evidence_reason_lower
            for term in ["weak", "partial", "below support threshold", "ambiguous"]
        ):
            evidence_for = [
                f"Claim has {len(claim.candidate_chunk_ids)} candidate chunks but zero supporting chunks",
                f"Evidence reason indicates weak support: {claim.evidence_reason}",
            ]

            if claim.verification_method:
                evidence_for.append(f"Verification method: {claim.verification_method}")

            evidence_against = []
            if trace.sufficiency_result and not trace.sufficiency_result.sufficient:
                evidence_against.append(
                    "Sufficiency reports insufficient context, suggesting retrieval miss rather than weak evidence"
                )

            return CandidateCause(
                cause_id=f"{claim.claim_text[:50]}_weak_evidence",
                cause_type="weak_or_ambiguous_evidence",
                stage=FailureStage.GROUNDING,
                evidence_for=evidence_for,
                evidence_against=evidence_against,
                affected_claims=[claim.claim_text],
                affected_chunk_ids=list(claim.candidate_chunk_ids),
                supporting_analyzers=["ClaimGroundingAnalyzer"],
                contradicting_analyzers=[],
                abduct=(
                    "Claim has related chunks but evidence quality is weak/ambiguous, "
                    "indicating noisy retrieval or partial context."
                ),
                act="Tighten evidence matching thresholds or filter noisy candidates before grounding.",
                predict=(
                    "Stronger evidence filtering should reduce unsupported claims driven by ambiguous context."
                ),
                predicted_fix_effect="would_partially_fix",
                calibration_status="uncalibrated",
            )

    return None


def candidate_generation_contradicted(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate generation_contradicted_retrieved_evidence candidate if applicable."""
    if claim.label != "contradicted":
        return None

    if claim.contradicting_chunk_ids or (claim.value_conflicts and len(claim.value_conflicts) > 0):
        evidence_for = ["Claim label is 'contradicted'"]

        if claim.contradicting_chunk_ids:
            evidence_for.append(f"Contradicting chunks: {', '.join(claim.contradicting_chunk_ids)}")

        if claim.value_conflicts:
            evidence_for.append(f"Value conflicts detected: {len(claim.value_conflicts)} conflicts")
            for conflict in claim.value_conflicts[:2]:
                evidence_for.append(f"  - {conflict}")

        if claim.evidence_reason:
            evidence_for.append(f"Evidence reason: {claim.evidence_reason}")

        evidence_against = []
        if trace.avg_score < 0.6:
            evidence_against.append(
                f"Low retrieval scores ({trace.avg_score:.2f}) suggest poor quality evidence"
            )

        return CandidateCause(
            cause_id=f"{claim.claim_text[:50]}_generation_contradicted",
            cause_type="generation_contradicted_retrieved_evidence",
            stage=FailureStage.GENERATION,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            affected_claims=[claim.claim_text],
            affected_chunk_ids=list(claim.contradicting_chunk_ids),
            supporting_analyzers=["ClaimGroundingAnalyzer"],
            contradicting_analyzers=[],
            abduct=(
                "Claim contradicts retrieved evidence (value conflicts/contradicting chunks), "
                "pointing to generation-stage drift from context."
            ),
            act="Enforce contradiction-aware decoding, require chunk-backed value extraction.",
            predict=(
                "With stricter context-grounded generation, contradicted claims should be corrected."
            ),
            predicted_fix_effect="would_likely_fix",
            calibration_status="uncalibrated",
        )

    return None


def candidate_stale_source(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate stale_source_usage candidate if applicable."""
    if not trace.has_stale_retrieval:
        return None

    claim_chunks = set(
        claim.supporting_chunk_ids + claim.candidate_chunk_ids + claim.contradicting_chunk_ids
    )
    stale_chunk_overlap = any(
        chunk_id for chunk_id in claim_chunks if any(stale_doc in chunk_id for stale_doc in trace.stale_doc_ids)
    )

    if stale_chunk_overlap or not claim_chunks:
        evidence_for = ["StaleRetrievalAnalyzer flagged stale sources"]

        if trace.stale_doc_ids:
            evidence_for.append(f"Stale documents: {', '.join(trace.stale_doc_ids)}")

        if trace.stale_evidence:
            evidence_for.extend(trace.stale_evidence[:2])

        evidence_against = []
        if not stale_chunk_overlap and claim_chunks:
            evidence_against.append("Claim's specific chunks may not overlap with stale sources")

        return CandidateCause(
            cause_id=f"{claim.claim_text[:50]}_stale_source",
            cause_type="stale_source_usage",
            stage=FailureStage.RETRIEVAL,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            affected_claims=[claim.claim_text],
            affected_chunk_ids=list(claim_chunks),
            supporting_analyzers=["StaleRetrievalAnalyzer"],
            contradicting_analyzers=[],
            abduct=(
                "Retrieved evidence comes from stale/outdated sources, "
                "meaning even if claim is textually supported, evidence may be superseded."
            ),
            act="Apply freshness filtering, prioritize latest document versions.",
            predict=(
                "Using current sources should change claim support if facts have changed."
            ),
            predicted_fix_effect="would_partially_fix",
            calibration_status="uncalibrated",
        )

    return None


def candidate_citation_mismatch(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate citation_mismatch candidate if applicable."""
    if not trace.has_citation_mismatch:
        return None

    evidence_for = ["CitationMismatchAnalyzer detected phantom citations"]

    if trace.phantom_citations:
        evidence_for.append(f"Phantom citations: {', '.join(trace.phantom_citations)}")

    evidence_against = []
    if not trace.cited_doc_ids:
        evidence_against.append("No citations provided, so mismatch may not affect this claim")

    return CandidateCause(
        cause_id=f"{claim.claim_text[:50]}_citation_mismatch",
        cause_type="citation_mismatch",
        stage=FailureStage.RETRIEVAL,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        affected_claims=[claim.claim_text],
        affected_chunk_ids=[],
        supporting_analyzers=["CitationMismatchAnalyzer"],
        contradicting_analyzers=[],
        abduct=(
            "Answer cites documents not in retrieved context, "
            "indicating retrieval miss or hallucinated citations."
        ),
        act="Require citations to map to retrieved chunk IDs or retrieve cited documents explicitly.",
        predict=(
            "Citation repair would make provenance valid, but may not fix claim truth. "
            "This is governance/traceability fix."
        ),
        predicted_fix_effect="would_partially_fix",
        calibration_status="uncalibrated",
    )


def candidate_post_rationalized_citation(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate post_rationalized_citation candidate if applicable."""
    if not trace.has_post_rationalized_citation:
        return None

    evidence_for = ["CitationFaithfulnessProbe detected low-quality citation-claim alignment"]

    if trace.citation_probe_results:
        failed_probes = [
            p for p in trace.citation_probe_results if isinstance(p, dict) and not p.get("passed", True)
        ]
        evidence_for.append(f"{len(failed_probes)} probe(s) failed")

    return CandidateCause(
        cause_id=f"{claim.claim_text[:50]}_post_rationalized",
        cause_type="post_rationalized_citation",
        stage=FailureStage.GROUNDING,
        evidence_for=evidence_for,
        evidence_against=[],
        affected_claims=[claim.claim_text],
        affected_chunk_ids=[],
        supporting_analyzers=["CitationFaithfulnessProbe"],
        contradicting_analyzers=[],
        abduct=(
            "Citations appear post-rationalized: answer may have been generated from parametric memory "
            "with citations attached afterward."
        ),
        act="Run counterfactual test: perturb cited source and check if answer changes.",
        predict=(
            "If citation is post-rationalized, removing it should not change answer."
        ),
        predicted_fix_effect="would_partially_fix",
        calibration_status="uncalibrated",
    )


def candidate_verification_uncertainty(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate verification_uncertainty candidate if applicable."""
    if not claim.fallback_used:
        return None

    evidence_for = [
        "Claim verification used fallback method",
        f"Verification method: {claim.verification_method or 'unknown'}",
    ]

    if claim.evidence_reason:
        evidence_for.append(f"Evidence reason: {claim.evidence_reason}")

    return CandidateCause(
        cause_id=f"{claim.claim_text[:50]}_verification_uncertainty",
        cause_type="verification_uncertainty",
        stage=FailureStage.GROUNDING,
        evidence_for=evidence_for,
        evidence_against=[],
        affected_claims=[claim.claim_text],
        affected_chunk_ids=list(claim.supporting_chunk_ids + claim.candidate_chunk_ids),
        supporting_analyzers=["ClaimGroundingAnalyzer"],
        contradicting_analyzers=[],
        abduct=(
            "Claim grounding used fallback heuristic due to verifier failure, "
            "introducing uncertainty about claim label accuracy."
        ),
        act="Deploy stronger verifier (LLM judge, NLI model) or request human review.",
        predict=(
            "Stronger verification may change claim label or reduce uncertainty."
        ),
        predicted_fix_effect="unknown",
        calibration_status="uncalibrated",
    )


def candidate_security_adversarial(claim: ClaimResult, trace: AttributionTrace) -> CandidateCause | None:
    """Generate prompt_injection_or_adversarial_context candidate if applicable."""
    if not trace.has_security_failure:
        return None

    evidence_for = [f"{len(trace.security_failure_types)} security failure(s) detected"]

    for failure_type in trace.security_failure_types:
        evidence_for.append(f"  - {failure_type.value}")

    if trace.security_evidence:
        evidence_for.extend(trace.security_evidence[:2])

    evidence_against = []
    if claim.label == "entailed":
        evidence_against.append(
            "Claim is entailed, suggesting adversarial content may not have compromised this claim"
        )

    return CandidateCause(
        cause_id=f"{claim.claim_text[:50]}_security_adversarial",
        cause_type="adversarial_context",
        stage=FailureStage.SECURITY,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        affected_claims=[claim.claim_text],
        affected_chunk_ids=[],
        supporting_analyzers=["PromptInjectionAnalyzer", "RetrievalAnomalyAnalyzer"],
        contradicting_analyzers=[],
        abduct=(
            "Security analyzers detected adversarial context or prompt injection, "
            "which may have influenced answer generation."
        ),
        act="Remove/sanitize suspicious chunks, isolate adversarial documents.",
        predict=(
            "If adversarial chunk caused failure, removing it should change answer/risk state."
        ),
        predicted_fix_effect="unknown",
        calibration_status="uncalibrated",
    )
