"""A2P (Abduct-Act-Predict) counterfactual attribution analyzer."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.attribution.trace import extract_attribution_trace
from raggov.analyzers.attribution.candidates import (
    identify_claims_needing_attribution,
    generate_candidate_causes,
)
from raggov.analyzers.attribution.scoring import score_all_candidates
from raggov.analyzers.attribution.selection import (
    select_primary_and_secondary_causes,
    build_evidence_summary,
    build_composite_fix_recommendation,
)
from raggov.models.diagnosis import (
    AnalyzerResult,
    CandidateCause,
    ClaimAttribution,
    ClaimAttributionV2,
    ClaimResult,
    FailureStage,
    FailureType,
    SufficiencyResult,
)

if TYPE_CHECKING:
    from raggov.models.grounding import GroundingEvidenceBundle, ClaimEvidenceRecord
from raggov.models.run import RAGRun


DEFAULT_LOW_CONFIDENCE = 0.45
DEFAULT_RETRIEVAL_CONFIDENCE = 0.58
RETRIEVAL_ANOMALY_CONFIDENCE = 0.82
LOW_SCORE_FEW_CHUNKS_CONFIDENCE = 0.78
LOW_SCORE_RETRIEVAL_CONFIDENCE = 0.65
INCONSISTENT_CHUNKS_CONFIDENCE = 0.68
PARSER_TABLE_CONFIDENCE = 0.86
PARSER_HIERARCHY_CONFIDENCE = 0.82
GENERATION_HIGH_CONFIDENCE = 0.74
GENERATION_MEDIUM_CONFIDENCE = 0.66

A2P_PROMPT_TEMPLATE = """
You are diagnosing a RAG pipeline failure. Use the A2P (Abduct-Act-Predict) framework:

OBSERVED FAILURES:
{failures}

RAG RUN CONTEXT:
- Query: {query}
- Retrieved chunks: {n_chunks} chunks, avg score: {avg_score:.2f}
- Answer length: {answer_length} chars

STEP 1 — ABDUCTION: What is the most likely hidden root cause in the pipeline
(PARSING/CHUNKING/EMBEDDING/RETRIEVAL/RERANKING/GENERATION/SECURITY) that explains ALL
observed failures? Reason through the causal chain.

STEP 2 — ACTION: What is the minimal corrective intervention at that stage
that would address the root cause?

STEP 3 — PREDICTION: If that intervention were applied, would the observed
failures be resolved? What residual failures might remain?

Respond ONLY with valid JSON:
{{
  "abduction": "<root cause reasoning>",
  "root_cause_stage": "<PARSING|CHUNKING|EMBEDDING|RETRIEVAL|RERANKING|GENERATION|SECURITY>",
  "action": "<specific fix>",
  "prediction": "<expected outcome after fix>",
  "confidence": <0.0-1.0>,
  "confidence_basis": "<why this confidence level>",
  "root_cause_type": "<optional FailureType enum value if you can determine it>"
}}
""".strip()

SECURITY_FAILURE_TYPES = {
    FailureType.PROMPT_INJECTION,
    FailureType.SUSPICIOUS_CHUNK,
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.PRIVACY_VIOLATION,
}


class A2PAttributionAnalyzer(BaseAnalyzer):
    """
    Implements the Abduct-Act-Predict (A2P) framework for counterfactual failure attribution.

    Given a failed RAG run and prior analyzer results, it reasons through which pipeline
    stage most likely caused the failure using structured counterfactual reasoning.

    A2P works in three steps:
    - Abduction: Infer the hidden root cause behind the observed failure
    - Action: Define the minimal corrective intervention at that stage
    - Prediction: Simulate whether the fix would resolve the failure
    """

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        """Analyze a RAG run and attribute failure to root cause stage."""
        prior_results = self.config.get("weighted_prior_results") or self.config.get("prior_results", [])

        failed_results = [
            r for r in prior_results if r.status == "fail" and r.failure_type is not None
        ]
        if not failed_results:
            return self.skip("no prior failures to attribute")

        # Extract grounding bundle
        bundle = self._get_grounding_bundle(prior_results)

        # Check for v2 mode
        use_v2 = self.config.get("use_a2p_v2", False)
        if use_v2:
            claim_level_result_v2 = self._claim_level_mode_v2(run, prior_results, bundle=bundle)
            if claim_level_result_v2 is not None:
                return claim_level_result_v2

        # v1 claim-level mode (backward compatibility)
        claim_level_result = self._claim_level_mode(run, prior_results, bundle=bundle)
        if claim_level_result is not None:
            return claim_level_result

        use_llm = self.config.get("use_llm", False)
        llm_fn = self.config.get("llm_fn")

        if use_llm and llm_fn:
            return self._llm_mode(run, prior_results, llm_fn)
        else:
            return self._attach_legacy_claim_fallback(
                self._deterministic_mode(run, prior_results)
            )

    def _claim_level_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult], bundle: GroundingEvidenceBundle | None = None
    ) -> AnalyzerResult | None:
        claim_results = self._claim_results_from_prior(prior_results)
        attributed_claims = [
            claim
            for claim in claim_results
            if self._claim_needs_attribution(claim, prior_results)
        ]
        if not attributed_claims:
            return None

        sufficiency_result = self._sufficiency_from_prior(prior_results)
        failure_types_present = {
            result.failure_type
            for result in prior_results
            if result.status in {"fail", "warn"} and result.failure_type is not None
        }
        has_citation_mismatch = FailureType.CITATION_MISMATCH in failure_types_present
        has_stale_retrieval = FailureType.STALE_RETRIEVAL in failure_types_present
        has_post_rationalized_citation = FailureType.POST_RATIONALIZED_CITATION in failure_types_present or any(
            result.analyzer_name == "CitationFaithfulnessProbe" and result.status in {"fail", "warn"}
            for result in prior_results
        )
        has_security_failure = any(
            result.failure_type in SECURITY_FAILURE_TYPES and result.status in {"fail", "warn"}
            for result in prior_results
        )

        # Map records by claim text for lookup
        records_by_text = {r.claim_text: r for r in bundle.claim_evidence_records} if bundle else {}

        attributions: list[ClaimAttribution] = []
        for claim in attributed_claims:
            record = records_by_text.get(claim.claim_text)
            attribution = self._attribute_claim(
                claim=claim,
                record=record,
                sufficiency_result=sufficiency_result,
                has_citation_mismatch=has_citation_mismatch,
                has_stale_retrieval=has_stale_retrieval,
                has_post_rationalized_citation=has_post_rationalized_citation,
                has_security_failure=has_security_failure,
            )
            attributions.append(attribution)

        strongest = self._strongest_claim_attribution(attributions)
        top_failure_type, top_stage, top_fix, top_conf = self._map_primary_cause_to_top_level(
            strongest.primary_cause
        )
        summary = (
            f"Claim-level A2P ({strongest.attribution_method}) primary cause: "
            f"{strongest.primary_cause} for claim '{strongest.claim_text}'. "
            f"Predicted intervention effect: {strongest.predict}"
        )
        evidence = [
            summary,
            f"Proposed fix: {top_fix}",
            f"Prediction: {strongest.predict}",
            "Confidence basis: Heuristic claim-level attribution (uncalibrated).",
        ]

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=top_failure_type,
            stage=top_stage,
            evidence=evidence,
            claim_attributions=attributions,
            remediation=top_fix,
            attribution_stage=top_stage,
            proposed_fix=top_fix,
            fix_confidence=top_conf,
            score=top_conf,
        )

    def _claim_level_mode_v2(
        self, run: RAGRun, prior_results: list[AnalyzerResult], bundle: GroundingEvidenceBundle | None = None
    ) -> AnalyzerResult | None:
        """A2P v2: Multi-hypothesis attribution with transparent scoring and primary/secondary selection."""
        # Step 1: Extract structured trace
        trace = extract_attribution_trace(run, prior_results)

        # Step 2: Identify claims needing attribution (failed or risky)
        claims_needing_attribution = identify_claims_needing_attribution(trace)
        if not claims_needing_attribution:
            return None

        # Step 3: Generate attributions for each claim
        attributions_v2: list[ClaimAttributionV2] = []
        for claim, reason in claims_needing_attribution:
            # Generate candidate causes
            candidates = generate_candidate_causes(claim, reason, trace)
            if not candidates:
                continue

            # Score candidates
            scored_candidates = score_all_candidates(candidates)

            # Select primary and secondary causes
            primary_cause, secondary_causes = select_primary_and_secondary_causes(scored_candidates)

            # Find primary candidate object
            primary_candidate = next((c for c in scored_candidates if c.cause_type == primary_cause), None)
            if not primary_candidate:
                continue

            # Find secondary candidate objects
            secondary_candidates = [c for c in scored_candidates if c.cause_type in secondary_causes]

            # Build evidence summary
            evidence_summary = build_evidence_summary(primary_candidate, secondary_candidates)

            # Build composite fix recommendation
            recommended_fix, recommended_fix_category = build_composite_fix_recommendation(
                primary_candidate, secondary_candidates
            )

            # Create ClaimAttributionV2
            attribution_v2 = ClaimAttributionV2(
                claim_text=claim.claim_text,
                claim_label=claim.label,
                primary_cause=primary_cause,
                secondary_causes=secondary_causes,
                candidate_causes=scored_candidates,
                evidence_summary=evidence_summary,
                recommended_fix=recommended_fix,
                recommended_fix_category=recommended_fix_category,
                attribution_method="claim_level_counterfactual_a2p_v2",
                fallback_used=False,
                calibration_status="uncalibrated",
            )
            attributions_v2.append(attribution_v2)

        if not attributions_v2:
            return None

        # Step 4: Select strongest attribution for top-level output
        strongest = self._strongest_claim_attribution_v2(attributions_v2)
        top_failure_type, top_stage, top_fix, top_conf = self._map_primary_cause_to_top_level(
            strongest.primary_cause
        )

        summary = (
            f"Claim-level A2P v2 ({strongest.attribution_method}) primary cause: "
            f"{strongest.primary_cause} for claim '{strongest.claim_text}'"
        )
        if strongest.secondary_causes:
            summary += f" (secondary: {', '.join(strongest.secondary_causes[:2])})"

        evidence = [summary] + strongest.evidence_summary[:5]

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=top_failure_type,
            stage=top_stage,
            evidence=evidence,
            claim_attributions_v2=attributions_v2,
            remediation=top_fix,
            attribution_stage=top_stage,
            proposed_fix=top_fix,
            fix_confidence=top_conf,
            score=top_conf,
        )

    def _strongest_claim_attribution_v2(
        self, attributions: list[ClaimAttributionV2]
    ) -> ClaimAttributionV2:
        """Select strongest attribution from v2 attributions by primary cause priority."""
        priority = {
            "generation_contradicted_retrieved_evidence": 0,
            "insufficient_context_or_retrieval_miss": 1,
            "weak_or_ambiguous_evidence": 2,
            "stale_source_usage": 3,
            "citation_mismatch": 4,
            "post_rationalized_citation": 5,
            "verification_uncertainty": 6,
            "adversarial_context": 7,
        }
        return sorted(
            attributions,
            key=lambda item: priority.get(item.primary_cause, 99),
        )[0]

    def _attribute_claim(
        self,
        claim: ClaimResult,
        record: ClaimEvidenceRecord | None,
        sufficiency_result: SufficiencyResult | None,
        has_citation_mismatch: bool,
        has_stale_retrieval: bool,
        has_post_rationalized_citation: bool,
        has_security_failure: bool,
    ) -> ClaimAttribution:
        candidate_causes: list[str] = []
        evidence = [claim.evidence_reason] if claim.evidence_reason else []
        affected_chunk_ids = sorted(
            {
                *claim.supporting_chunk_ids,
                *claim.candidate_chunk_ids,
                *claim.contradicting_chunk_ids,
            }
        )

        if record and record.uncertainty_signals.get("value_conflicts"):
            primary_cause = "value_distortion"
            abduct = (
                f"Claim '{claim.claim_text}' has direct value conflicts with retrieved evidence "
                f"(e.g., mismatched dates, numbers, or entities), pointing to generation-stage hallucination."
            )
            act = "Enforce stricter value-level grounding and verify all numerical/named entities against source context."
            predict = "Fixing value-level hallucination is predicted to resolve the contradiction while preserving the claim structure."
        elif claim.label == "unsupported" and record and not record.supporting_chunk_ids and record.candidate_evidence_chunks:
            # Check if candidates were high-scoring but failed verification
            max_candidate_score = max(
                (
                    getattr(c, "score", None)
                    if getattr(c, "score", None) is not None
                    else getattr(c, "raw_support_score", 0.0)
                    for c in record.candidate_evidence_chunks
                ),
                default=0.0,
            )
            if max_candidate_score > 0.7: # Heuristic threshold for "it was there but ignored"
                primary_cause = "context_ignored"
                abduct = (
                    "High-scoring candidate evidence was found, but the generator failed to utilize it correctly "
                    "or the verifier rejected the alignment. Likely context-ignoring failure."
                )
                act = "Optimize generator prompt to improve context adherence and utilization of identified chunks."
                predict = "Improved context adherence is predicted to convert this unsupported claim into an entailed one."
            else:
                primary_cause = "insufficient_context_or_retrieval_miss"
                abduct = (
                    "No supporting chunks found and candidate evidence is weak or irrelevant, "
                    "pointing to a retrieval miss or missing information in the knowledge base."
                )
                act = "Increase retrieval recall and verify if the knowledge base contains the required information."
                predict = "Improving retrieval recall is predicted to surface the missing evidence needed for this claim."
        elif claim.label == "unsupported" and not claim.supporting_chunk_ids and (
            sufficiency_result is not None and sufficiency_result.sufficient is False
        ):
            primary_cause = "insufficient_context_or_retrieval_miss"
            abduct = (
                "Claim is unsupported with no supporting chunks while sufficiency indicates missing evidence; "
                "the likely hidden cause is retrieval/context insufficiency."
            )
            act = "Increase retrieval depth and query recall to surface missing supporting evidence."
            predict = (
                "If retrieval/context is improved, this claim is expected to become verifiable "
                "or be correctly abstained."
            )
        elif claim.label == "contradicted" and claim.contradicting_chunk_ids:
            primary_cause = "generation_contradicted_retrieved_evidence"
            abduct = (
                "Claim contradicts retrieved evidence in identified chunks, pointing to generation-stage drift from context."
            )
            act = "Enforce context-grounded generation with contradiction-aware decoding/citation checks."
            predict = (
                "With stricter grounded generation, contradicted claims are expected to be removed or corrected."
            )
        # ... rest of the legacy cases ...
        elif claim.label == "entailed" and has_security_failure:
            primary_cause = "adversarial_context"
            abduct = (
                "Claim is textually entailed, but security analyzers detected adversarial or poisoned context, "
                "so the governance risk is that unsafe context influenced the answer path."
            )
            act = "Remove or quarantine adversarial chunks and require security filtering before answer generation."
            predict = (
                "If adversarial context is removed, the answer should remain supported without inheriting security risk."
            )
        elif claim.label == "entailed" and has_citation_mismatch:
            primary_cause = "citation_mismatch"
            abduct = (
                "Claim is textually entailed, but citations do not map to retrieved evidence, indicating provenance failure."
            )
            act = "Require citations to resolve to retrieved chunk/document IDs before returning the answer."
            predict = (
                "Repairing citation provenance should preserve claim truth while fixing governance and traceability."
            )
        elif claim.label == "entailed" and has_post_rationalized_citation:
            primary_cause = "post_rationalized_citation"
            abduct = (
                "Claim is textually entailed, but citation faithfulness signals imply citations were attached after generation."
            )
            act = "Enforce faithfulness checks so cited evidence must causally support the generated answer."
            predict = (
                "If post-rationalized citations are blocked, the answer should either cite valid support or be revised."
            )
        elif claim.label == "entailed" and has_stale_retrieval:
            primary_cause = "stale_source_usage"
            abduct = (
                "Claim is textually entailed by retrieved evidence, but that evidence is stale and may no longer be governing."
            )
            act = "Filter or down-rank stale sources and prefer current document versions during retrieval."
            predict = (
                "Using fresh sources should preserve valid claims and surface superseded ones for correction."
            )
        else:
            primary_cause = "weak_or_ambiguous_evidence"
            abduct = (
                "Failed claim cannot be cleanly isolated; weak evidence linkage is the most plausible claim-level cause."
            )
            act = "Strengthen evidence matching and require explicit support checks before finalizing the claim."
            predict = "Improved evidence checks are predicted to reduce unresolved failed-claim cases."

        if has_citation_mismatch:
            candidate_causes.append("citation_mismatch")
        if has_post_rationalized_citation:
            candidate_causes.append("post_rationalized_citation")
        if has_stale_retrieval:
            candidate_causes.append("stale_source_usage")
        if has_security_failure:
            candidate_causes.append("adversarial_context")
        if claim.fallback_used:
            candidate_causes.append("verification_uncertainty")
        if primary_cause not in candidate_causes:
            candidate_causes.insert(0, primary_cause)

        return ClaimAttribution(
            claim_text=claim.claim_text,
            claim_label=claim.label,
            candidate_causes=candidate_causes,
            primary_cause=primary_cause,
            abduct=abduct,
            act=act,
            predict=predict,
            evidence=evidence,
            affected_chunk_ids=affected_chunk_ids,
            attribution_method="claim_level_a2p_heuristic_v1",
            calibration_status="uncalibrated",
            fallback_used=False,
        )

    def _strongest_claim_attribution(
        self, attributions: list[ClaimAttribution]
    ) -> ClaimAttribution:
        priority = {
            "generation_contradicted_retrieved_evidence": 0,
            "insufficient_context_or_retrieval_miss": 1,
            "weak_or_ambiguous_evidence": 2,
            "stale_source_usage": 3,
            "citation_mismatch": 4,
            "post_rationalized_citation": 5,
            "verification_uncertainty": 6,
            "adversarial_context": 7,
        }
        return sorted(
            attributions,
            key=lambda item: priority.get(item.primary_cause, 99),
        )[0]

    def _map_primary_cause_to_top_level(
        self, primary_cause: str
    ) -> tuple[FailureType, FailureStage, str, float]:
        if primary_cause == "generation_contradicted_retrieved_evidence":
            return (
                FailureType.GENERATION_IGNORE,
                FailureStage.GENERATION,
                "Enforce claim-level contradiction checks during generation and require chunk-backed support.",
                GENERATION_MEDIUM_CONFIDENCE,
            )
        if primary_cause == "insufficient_context_or_retrieval_miss":
            return (
                FailureType.RETRIEVAL_DEPTH_LIMIT,
                FailureStage.RETRIEVAL,
                "Increase retrieval depth and improve query recall to capture missing claim evidence.",
                DEFAULT_RETRIEVAL_CONFIDENCE,
            )
        if primary_cause == "stale_source_usage":
            return (
                FailureType.STALE_RETRIEVAL,
                FailureStage.RETRIEVAL,
                "Prefer current document versions and apply freshness-aware retrieval filtering.",
                DEFAULT_RETRIEVAL_CONFIDENCE,
            )
        if primary_cause == "citation_mismatch":
            return (
                FailureType.CITATION_MISMATCH,
                FailureStage.RETRIEVAL,
                "Require answer citations to resolve to retrieved evidence before returning the response.",
                DEFAULT_RETRIEVAL_CONFIDENCE,
            )
        if primary_cause == "post_rationalized_citation":
            return (
                FailureType.POST_RATIONALIZED_CITATION,
                FailureStage.GROUNDING,
                "Block post-rationalized citations by verifying that cited evidence causally supports the answer.",
                DEFAULT_LOW_CONFIDENCE,
            )
        if primary_cause == "adversarial_context":
            return (
                FailureType.PROMPT_INJECTION,
                FailureStage.SECURITY,
                "Sanitize or quarantine adversarial retrieved content before answer generation.",
                DEFAULT_RETRIEVAL_CONFIDENCE,
            )
        return (
            FailureType.UNSUPPORTED_CLAIM,
            FailureStage.GROUNDING,
            "Strengthen evidence selection and disallow weakly supported claims in final answers.",
            DEFAULT_LOW_CONFIDENCE,
        )

    def _claim_needs_attribution(
        self, claim: ClaimResult, prior_results: list[AnalyzerResult]
    ) -> bool:
        if claim.label in {"unsupported", "contradicted"}:
            return True
        if claim.label != "entailed":
            return False

        for result in prior_results:
            if result.status not in {"fail", "warn"}:
                continue
            if result.failure_type in {
                FailureType.CITATION_MISMATCH,
                FailureType.STALE_RETRIEVAL,
                FailureType.POST_RATIONALIZED_CITATION,
            }:
                return True
            if result.analyzer_name == "CitationFaithfulnessProbe":
                return True
            if result.failure_type in SECURITY_FAILURE_TYPES:
                return True
        return claim.fallback_used

    def _claim_results_from_prior(self, prior_results: list[AnalyzerResult]) -> list[ClaimResult]:
        for result in prior_results:
            if result.analyzer_name == "ClaimGroundingAnalyzer" and result.claim_results:
                return result.claim_results
        return []

    def _sufficiency_from_prior(
        self, prior_results: list[AnalyzerResult]
    ) -> SufficiencyResult | None:
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

    def _deterministic_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult]
    ) -> AnalyzerResult:
        """Deterministic A2P approximation using causal rules over the pipeline DAG."""
        analyzer_weights = self.config.get("analyzer_weights", {})
        best_results = self._best_results_by_failure(prior_results, analyzer_weights)
        failure_types = {
            result.failure_type
            for result in prior_results
            if result.status == "fail" and result.failure_type is not None
        }
        chunk_scores = [c.score for c in run.retrieved_chunks if c.score is not None]
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        n_chunks = len(chunk_scores)
        candidates: list[tuple[float, int, AnalyzerResult]] = []

        table_result = best_results.get(FailureType.TABLE_STRUCTURE_LOSS)
        if table_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.TABLE_STRUCTURE_LOSS,
                stage=FailureStage.PARSING,
                abduction=(
                    "Observed table-structure loss is most consistent with a parser-stage "
                    "failure: row and column bindings were stripped before chunking could preserve them."
                ),
                action=table_result.remediation or self._remediation_for(prior_results, FailureType.TABLE_STRUCTURE_LOSS),
                prediction=(
                    "Restoring table-aware parsing should propagate intact cell bindings into chunking "
                    "and prevent the downstream loss of grounded table facts."
                ),
                confidence=PARSER_TABLE_CONFIDENCE,
                confidence_basis=(
                    "High confidence because TABLE_STRUCTURE_LOSS has a direct causal predecessor in the DAG: "
                    "parsing errors occur before chunking and manifest with this exact structure-loss symptom."
                ),
            )
            candidates.append(self._candidate(table_result, result, analyzer_weights, 0))

        hierarchy_result = best_results.get(FailureType.HIERARCHY_FLATTENING)
        if hierarchy_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.HIERARCHY_FLATTENING,
                stage=FailureStage.PARSING,
                abduction=(
                    "Hierarchy flattening points upstream to parsing because heading and list relationships "
                    "must be lost before chunking or retrieval can act on the document."
                ),
                action=hierarchy_result.remediation or self._remediation_for(prior_results, FailureType.HIERARCHY_FLATTENING),
                prediction=(
                    "Preserving headings and numbered-list structure should keep related content grouped "
                    "and reduce downstream scope and grounding errors."
                ),
                confidence=PARSER_HIERARCHY_CONFIDENCE,
                confidence_basis=(
                    "High confidence because HIERARCHY_FLATTENING is primarily caused by parser output "
                    "that collapses document structure before later stages run."
                ),
            )
            candidates.append(self._candidate(hierarchy_result, result, analyzer_weights, 1))

        anomaly_result = best_results.get(FailureType.RETRIEVAL_ANOMALY)
        if FailureType.RETRIEVAL_ANOMALY in failure_types and anomaly_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.EMBEDDING_DRIFT,
                stage=FailureStage.EMBEDDING,
                abduction=(
                    "Score anomalies in the retrieved set indicate embedding-space corruption: "
                    "semantically dissimilar documents were mapped into a high-similarity region."
                ),
                action="Re-evaluate the embedding model on-domain and audit the vector index for corruption.",
                prediction=(
                    "Fixing the embedding representation should eliminate anomalous nearest neighbors "
                    "and reduce downstream grounding and retrieval instability."
                ),
                confidence=RETRIEVAL_ANOMALY_CONFIDENCE,
                confidence_basis=(
                    "High confidence because RETRIEVAL_ANOMALY has one dominant causal predecessor in the DAG: "
                    "embedding/index quality, not answer generation or retrieval depth."
                ),
            )
            candidates.append(self._candidate(anomaly_result, result, analyzer_weights, 2))

        insufficient_result = best_results.get(FailureType.INSUFFICIENT_CONTEXT)
        if FailureType.INSUFFICIENT_CONTEXT in failure_types and insufficient_result is not None:
            if avg_score < 0.6 and n_chunks < 4:
                confidence = LOW_SCORE_FEW_CHUNKS_CONFIDENCE
                abduction = (
                    "Insufficient context plus uniformly low retrieval scores and a shallow candidate set "
                    "is best explained by retrieval depth being too small to surface the needed evidence."
                )
                action = "Increase top-k retrieval to at least 5-8 and inspect whether missing supporting chunks appear."
                prediction = (
                    "Expanding retrieval depth should resolve the insufficiency if the missing evidence exists in the corpus; "
                    "residual failure would point to query formulation or indexing."
                )
                confidence_basis = (
                    "Confidence 0.78 because both signals are present: low average similarity and fewer than four scored chunks, "
                    "which strongly narrows the cause to top-k depth rather than generation."
                )
            elif avg_score < 0.6:
                confidence = LOW_SCORE_RETRIEVAL_CONFIDENCE
                abduction = (
                    "Insufficient context with broadly weak retrieval scores suggests the retriever is missing relevant documents "
                    "rather than the generator ignoring good context."
                )
                action = "Audit retrieval recall, expand candidate depth, and verify query rewriting or domain indexing."
                prediction = (
                    "Improving retrieval recall should increase evidence coverage, though residual issues may remain if query scope is mismatched."
                )
                confidence_basis = (
                    "Confidence 0.65 because low retrieval scores support a retrieval-stage cause, but the signal does not isolate "
                    "whether the problem is depth, recall, or query formulation."
                )
            else:
                confidence = DEFAULT_RETRIEVAL_CONFIDENCE
                abduction = (
                    "Insufficient context despite non-trivial retrieval scores suggests the retriever is returning context adjacent to the query "
                    "without capturing the precise scope required to answer it."
                )
                action = "Refine query rewriting and retrieval filters to better match the requested scope before reranking."
                prediction = (
                    "Sharper retrieval scope should reduce insufficiency, but residual ambiguity may remain if the answer is absent from the corpus."
                )
                confidence_basis = (
                    "Confidence 0.58 because insufficient context alone does not isolate a single root cause, but retrieval remains the best fit in the DAG."
                )

            result = self._create_attribution_result(
                root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
                stage=FailureStage.RETRIEVAL,
                abduction=abduction,
                action=action,
                prediction=prediction,
                confidence=confidence,
                confidence_basis=confidence_basis,
            )
            candidates.append(self._candidate(insufficient_result, result, analyzer_weights, 3))

        inconsistent_result = best_results.get(FailureType.INCONSISTENT_CHUNKS)
        if FailureType.INCONSISTENT_CHUNKS in failure_types and inconsistent_result is not None:
            result = self._create_attribution_result(
                root_cause_type=FailureType.CHUNKING_BOUNDARY_ERROR,
                stage=FailureStage.CHUNKING,
                abduction=(
                    "Conflicting retrieved chunks are most plausibly caused by boundary errors that split or merge logical units "
                    "before retrieval, creating partial context fragments."
                ),
                action="Adjust chunk boundaries to preserve paragraph, section, and list cohesion.",
                prediction=(
                    "Cleaner chunk boundaries should reduce contradictory partial context, though residual issues may remain if parsing already lost structure."
                ),
                confidence=INCONSISTENT_CHUNKS_CONFIDENCE,
                confidence_basis=(
                    "Confidence 0.68 because inconsistent chunks are consistent with chunking errors, but parsing defects can create a similar symptom."
                ),
            )
            candidates.append(self._candidate(inconsistent_result, result, analyzer_weights, 4))

        unsupported_result = best_results.get(FailureType.UNSUPPORTED_CLAIM)
        if FailureType.UNSUPPORTED_CLAIM in failure_types and unsupported_result is not None and avg_score > 0.7:
            confidence = GENERATION_HIGH_CONFIDENCE if avg_score > 0.85 and n_chunks >= 2 else GENERATION_MEDIUM_CONFIDENCE
            confidence_basis = (
                "Confidence 0.74 because unsupported claims appear despite multiple strong retrieval scores, "
                "which is strong evidence that generation ignored available context."
                if confidence == GENERATION_HIGH_CONFIDENCE
                else "Confidence 0.66 because high-scoring context points toward generation-stage drift, but retrieval quality is not overwhelmingly strong."
            )
            result = self._create_attribution_result(
                root_cause_type=FailureType.GENERATION_IGNORE,
                stage=FailureStage.GENERATION,
                abduction=(
                    "Unsupported claims despite strong retrieved evidence indicate the model likely generated from parametric memory "
                    "or ignored the provided context during answer synthesis."
                ),
                action="Strengthen grounding instructions, require chunk-quoted evidence, or switch to a more context-adherent model.",
                prediction=(
                    "A stricter generation policy should resolve unsupported claims if retrieval remains strong; residual errors would suggest grounding heuristics still miss key evidence."
                ),
                confidence=confidence,
                confidence_basis=confidence_basis,
            )
            candidates.append(self._candidate(unsupported_result, result, analyzer_weights, 5))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]

        return self._create_attribution_result(
            root_cause_type=FailureType.RETRIEVAL_DEPTH_LIMIT,
            stage=FailureStage.RETRIEVAL,
            abduction=(
                "Observed failures do not isolate a single upstream cause, so retrieval remains the lowest-risk default hypothesis "
                "in the causal DAG because many downstream failures originate from missing evidence."
            ),
            action="Inspect retrieval recall first: verify top-k depth, query rewriting, and corpus coverage before changing downstream stages.",
            prediction=(
                "This intervention may surface missing evidence, but the weak signal means residual failures are likely and further diagnosis may still be required."
            ),
            confidence=DEFAULT_LOW_CONFIDENCE,
            confidence_basis=(
                "Confidence 0.45 because the current evidence does not distinguish among multiple upstream causes; this is a low-confidence fallback, not a precise attribution."
            ),
        )

    def _llm_mode(
        self, run: RAGRun, prior_results: list[AnalyzerResult], llm_fn: Callable[[str], str]
    ) -> AnalyzerResult:
        """LLM-based mode using an explicit Abduct-Act-Predict prompt."""
        prompt = self._build_a2p_prompt(run, prior_results)

        try:
            response = llm_fn(prompt)
            parsed = self._parse_llm_payload(response)
            root_cause_stage_str = parsed["root_cause_stage"]
            root_cause_stage = FailureStage(root_cause_stage_str)
            root_cause_type = self._infer_root_cause_type(
                root_cause_stage,
                prior_results,
                parsed.get("root_cause_type"),
            )
            return self._create_attribution_result(
                root_cause_type=root_cause_type,
                stage=root_cause_stage,
                abduction=str(parsed["abduction"]),
                action=str(parsed["action"]),
                prediction=str(parsed["prediction"]),
                confidence=float(parsed["confidence"]),
                confidence_basis=str(parsed["confidence_basis"]),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return self._attach_legacy_claim_fallback(self._deterministic_mode(run, prior_results))

    def _attach_legacy_claim_fallback(self, result: AnalyzerResult) -> AnalyzerResult:
        result.claim_attributions = [
            ClaimAttribution(
                claim_text="__legacy_failure_level_attribution__",
                claim_label="unknown",
                candidate_causes=["legacy_failure_level_heuristic"],
                primary_cause="legacy_failure_level_heuristic",
                abduct="Typed claim-level inputs unavailable; used legacy failure-level heuristic attribution.",
                act=result.proposed_fix or result.remediation or "Inspect prior analyzer failures.",
                predict="Applying the proposed fix is predicted to reduce observed failures, but claim-level effect is unverified.",
                evidence=list(result.evidence),
                affected_chunk_ids=[],
                attribution_method="legacy_failure_level_heuristic",
                calibration_status="uncalibrated",
                fallback_used=True,
            )
        ]
        return result

    def _build_a2p_prompt(self, run: RAGRun, prior_results: list[AnalyzerResult]) -> str:
        """Build the structured A2P prompt for LLM."""
        score_values = [chunk.score for chunk in run.retrieved_chunks if chunk.score is not None]
        avg_score = sum(score_values) / len(score_values) if score_values else 0.0
        return A2P_PROMPT_TEMPLATE.format(
            failures=self._format_failures(prior_results),
            query=run.query,
            n_chunks=len(run.retrieved_chunks),
            avg_score=avg_score,
            answer_length=len(run.final_answer),
        )

    def _remediation_for(
        self,
        prior_results: list[AnalyzerResult],
        failure_type: FailureType,
    ) -> str:
        """Return the concrete remediation attached to a prior result when available."""
        for result in prior_results:
            if result.failure_type == failure_type and result.remediation is not None:
                return result.remediation
        return ""

    def _best_results_by_failure(
        self,
        prior_results: list[AnalyzerResult],
        analyzer_weights: dict[str, float],
    ) -> dict[FailureType, AnalyzerResult]:
        best_results: dict[FailureType, tuple[float, int, AnalyzerResult]] = {}
        for order, result in enumerate(prior_results):
            if result.status != "fail" or result.failure_type is None:
                continue
            weight = self._result_weight(result, analyzer_weights)
            current = best_results.get(result.failure_type)
            if current is None or weight > current[0] or (weight == current[0] and order < current[1]):
                best_results[result.failure_type] = (weight, order, result)
        return {
            failure_type: payload[2]
            for failure_type, payload in best_results.items()
        }

    def _result_weight(
        self,
        result: AnalyzerResult,
        analyzer_weights: dict[str, float],
    ) -> float:
        return float(analyzer_weights.get(result.analyzer_name, 1.0))

    def _create_attribution_result(
        self,
        root_cause_type: FailureType,
        stage: FailureStage,
        abduction: str,
        action: str,
        prediction: str,
        confidence: float,
        confidence_basis: str,
    ) -> AnalyzerResult:
        """Create an attribution result with all fields populated."""
        evidence = [
            abduction,
            f"Proposed fix: {action}",
            f"Prediction: {prediction}",
            f"Confidence basis: {confidence_basis}",
        ]

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="fail",
            failure_type=root_cause_type,
            stage=stage,
            evidence=evidence,
            remediation=action,
            attribution_stage=stage,
            proposed_fix=action,
            fix_confidence=confidence,
            score=confidence,
        )

    def _candidate(
        self,
        source_result: AnalyzerResult,
        attribution_result: AnalyzerResult,
        analyzer_weights: dict[str, float],
        order: int,
    ) -> tuple[float, int, AnalyzerResult]:
        """Rank a deterministic attribution by evidence strength and analyzer weight."""
        weight = self._result_weight(source_result, analyzer_weights)
        confidence = attribution_result.fix_confidence or 0.0
        return (weight * confidence, order, attribution_result)

    def _format_failures(self, prior_results: list[AnalyzerResult]) -> str:
        """Format observed analyzer failures for the LLM prompt."""
        lines: list[str] = []
        for result in prior_results:
            if result.status not in {"fail", "warn"} or result.failure_type is None:
                continue
            snippet = "; ".join(result.evidence[:2]) if result.evidence else "no evidence provided"
            lines.append(
                f"- {result.failure_type.value} at {result.stage.value if result.stage else 'UNKNOWN'}: {snippet}"
            )
        return "\n".join(lines) if lines else "- No structured failures available"

    def _parse_llm_payload(self, response: Any) -> dict[str, Any]:
        """Normalize and validate an LLM A2P response payload."""
        if isinstance(response, dict):
            payload = response
        else:
            payload = json.loads(str(response))

        required_fields = {
            "abduction",
            "root_cause_stage",
            "action",
            "prediction",
            "confidence",
            "confidence_basis",
        }
        missing = required_fields - set(payload)
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")

        confidence = float(payload["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        payload["confidence"] = confidence
        return payload

    def _infer_root_cause_type(
        self,
        stage: FailureStage,
        prior_results: list[AnalyzerResult],
        declared_type: str | None = None,
    ) -> FailureType:
        """Infer a canonical root-cause failure type from stage and prior failures."""
        if declared_type:
            return FailureType(declared_type)

        if stage == FailureStage.PARSING:
            failure_types = {
                result.failure_type
                for result in prior_results
                if result.status == "fail" and result.failure_type is not None
            }
            if FailureType.TABLE_STRUCTURE_LOSS in failure_types:
                return FailureType.TABLE_STRUCTURE_LOSS
            if FailureType.HIERARCHY_FLATTENING in failure_types:
                return FailureType.HIERARCHY_FLATTENING
            return FailureType.PARSER_STRUCTURE_LOSS
        if stage == FailureStage.CHUNKING:
            return FailureType.CHUNKING_BOUNDARY_ERROR
        if stage == FailureStage.EMBEDDING:
            return FailureType.EMBEDDING_DRIFT
        if stage == FailureStage.RETRIEVAL:
            return FailureType.RETRIEVAL_DEPTH_LIMIT
        if stage == FailureStage.RERANKING:
            return FailureType.RERANKER_FAILURE
        if stage == FailureStage.GENERATION:
            return FailureType.GENERATION_IGNORE
        if stage == FailureStage.SECURITY:
            for result in prior_results:
                if result.status == "fail" and result.failure_type in SECURITY_FAILURE_TYPES:
                    return result.failure_type
            return FailureType.SUSPICIOUS_CHUNK
        return FailureType.RETRIEVAL_DEPTH_LIMIT

    def _get_grounding_bundle(self, prior_results: list[AnalyzerResult]) -> GroundingEvidenceBundle | None:
        """Extract the grounding bundle from prior results."""
        for result in prior_results:
            if getattr(result, "grounding_evidence_bundle", None):
                return result.grounding_evidence_bundle
        return None
