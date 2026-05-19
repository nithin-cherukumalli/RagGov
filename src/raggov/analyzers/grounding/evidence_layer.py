"""
Claim-Level Evidence Layer for GovRAG grounding analysis.

This is a v0 heuristic evidence layer. It is not yet an NLI verifier,
RefChecker implementation, RAGChecker implementation, or calibrated
confidence system.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from raggov.analyzers.grounding.claims import ExtractedClaim
from raggov.analyzers.grounding.value_extraction import (
    ValueMention,
    extract_value_mentions,
    find_value_alignment,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.grounding import (
    ClaimEvidenceRecord,
    ClaimVerificationLabel,
    CalibrationStatus,
    StructuredClaimRepresentation,
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate, EvidenceCandidateSelector
from raggov.analyzers.grounding.verifiers import EvidenceVerifier, TripletVerifier, VerificationResult
from raggov.analyzers.grounding.triplets import ClaimTriplet, TripletExtractor
from raggov.calibration.claim_calibration import ClaimCalibrationModel, CalibratedClaimConfidence, CalibrationMode


logger = logging.getLogger(__name__)

NEGATION_SIGNALS = {
    "not",
    "never",
    "no",
    "no longer",
    "contrary to",
    "prohibited",
    "forbidden",
    "disallowed",
    "not permitted",
}

_NUMERIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%|\bpercent\b|\bamount\b|\bthreshold\b|\bceiling\b|\blimit\b"
    r"|[$₹€£]|\brupees?\b|\brs\.?\b",
    re.IGNORECASE,
)
_VERSION_PATTERN = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b|\bversion\s+\d", re.IGNORECASE)
_DATE_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b|\bdeadline\b|\beffective\s+date\b",
    re.IGNORECASE,
)
_REQUIREMENT_PATTERN = re.compile(
    r"\beligib\w*|\bqualif\w*|\bentitl\w*|\bwho\s+can\b|\bwho\s+is\b"
    r"|\bmust\b|\bshall\b|\brequir(?:e|es|ed|ement)\b|\bcondition\b|\bconstraint\b",
    re.IGNORECASE,
)
_DEFINITION_PATTERN = re.compile(
    r"\bmeans?\b|\brefers?\b|\bis\s+defined\b|\bdenotes?\b",
    re.IGNORECASE,
)
_PROCEDURAL_PATTERN = re.compile(
    r"\bprocedure\b|\bsteps?\b|\bsection\b|\bmanual\b|\binstructions?\b|\bhow\s+to\b"
    r"|\breplace(?:ment)?\b|\binstall(?:ation)?\b|\bconfigure\b",
    re.IGNORECASE,
)
_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:supports?|deprecates?|replaces?|supersedes?|withdrawn|deprecated|causes?|"
    r"improves?|reduces?|increases?|decreases?|compared|baseline|higher|lower)\b",
    re.IGNORECASE,
)
_COMPOUND_CONJUNCTIONS = re.compile(
    r"\b(?:and|but|also|additionally|furthermore|moreover|while|whereas|however)\b",
    re.IGNORECASE,
)
_CITATION_REGEX = re.compile(r"\[(?:c|doc\s*)(\d+)\]", re.IGNORECASE)
_FINITE_VERBS = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|shall|must|should|can|may|"
    r"covers?|applies?|requires?|states?|provides?|allows?|prohibits?|entitles?)\b",
    re.IGNORECASE,
)








# Model imported from raggov.models.grounding


def detect_claim_type(claim: str) -> str:
    """
    Heuristic v0 claim type detection. Labels are not empirically validated.

    Priority: value/date/version > definition > requirement/procedure >
    relationship/comparison > general factual.
    """
    lowered = claim.lower()
    if _PROCEDURAL_PATTERN.search(claim):
        return "policy_rule"
    if re.search(r"\bsection\s+\d+(?:\.\d+)*\b", lowered):
        return "policy_rule"
    if _VERSION_PATTERN.search(claim):
        return "version_validity"
    if _NUMERIC_PATTERN.search(claim):
        return "numeric"
    if _DATE_PATTERN.search(claim):
        return "temporal"
    if _DEFINITION_PATTERN.search(claim):
        return "definition"
    if re.search(r"\bmust\b|\bshall\b", lowered):
        return "obligation"
    if re.search(r"\bmust not\b|\bmay not\b|\bprohibit", lowered):
        return "prohibition"
    if _REQUIREMENT_PATTERN.search(claim):
        return "eligibility"
    if re.search(r"\bcaus|because|due to|leads to\b", lowered):
        return "causal"
    if _RELATIONSHIP_PATTERN.search(claim):
        return "comparison"
    if re.search(r"\bpolicy\b|\bregulation\b|\border\b|\brule\b", lowered):
        return "policy_rule"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim):
        return "entity_attribute"
    return "other"


def detect_atomicity(claim: str) -> str:
    """
    Heuristic v0 atomicity detection. Labels are not empirically validated.

    Returns: atomic | compound | unclear
    """
    if len(claim.split()) < 5:
        return "unclear"
    if len(_COMPOUND_CONJUNCTIONS.findall(claim)) >= 2:
        return "compound"
    if len(_FINITE_VERBS.findall(claim)) > 2:
        return "compound"
    return "atomic"





class ClaimEvidenceBuilder:
    """
    Builds ClaimEvidenceRecord objects by coordinating ClaimExtractor and
    HeuristicValueOverlapVerifier. Entry point for the heuristic evidence layer.
    """

    def __init__(
        self,
        verifier: EvidenceVerifier,
        selector: EvidenceCandidateSelector,
        triplet_extractor: TripletExtractor | None = None,
        calibrator: ClaimCalibrationModel | None = None,
    ) -> None:
        self._verifier = verifier
        self._selector = selector
        self._triplet_extractor = triplet_extractor
        self._triplet_verifier = None
        self._calibrator = calibrator or ClaimCalibrationModel(CalibrationMode.NONE, {})

    def set_triplet_verifier(self, verifier: TripletVerifier) -> None:
        self._triplet_verifier = verifier

    def build(
        self,
        claims: list[ExtractedClaim | str],
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[ClaimEvidenceRecord]:
        return [
            self._build_single(claim, index, query, chunks)
            for index, claim in enumerate(claims, start=1)
        ]

    def _build_single(
        self,
        claim: ExtractedClaim | str,
        index: int,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> ClaimEvidenceRecord:
        structured_claim = claim if isinstance(claim, ExtractedClaim) else None
        claim_text = structured_claim.claim_text if structured_claim is not None else claim
        claim_id = structured_claim.claim_id if structured_claim is not None else f"claim_{index:03d}"
        candidates = self._selector.select_candidates(claim_text, query, chunks)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        citation_text = (
            structured_claim.source_sentence
            if structured_claim is not None and structured_claim.source_sentence
            else claim_text
        )
        cited_ids = []
        for match in _CITATION_REGEX.finditer(citation_text):
            cited_ids.append(f"c{match.group(1)}")
        cited_doc_ids = [
            chunks_by_id[chunk_id].source_doc_id
            for chunk_id in cited_ids
            if chunk_id in chunks_by_id
        ]

        if structured_claim is not None and not structured_claim.should_verify:
            return ClaimEvidenceRecord(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=structured_claim.source_sentence,
                source_answer_span=(structured_claim.source_start_char, structured_claim.source_end_char),
                claim_type=structured_claim.claim_type,
                atomicity_status=structured_claim.atomicity_status,
                entities=list(structured_claim.entities),
                dates=list(structured_claim.dates),
                numbers=list(structured_claim.numbers),
                extraction_method=structured_claim.extraction_method,
                extraction_reason=structured_claim.extraction_reason,
                extraction_confidence=structured_claim.extraction_confidence,
                extraction_warnings=list(structured_claim.extraction_warnings),
                skip_reason=structured_claim.skip_reason,
                extracted_values=extract_value_mentions(claim_text),
                cited_doc_ids=list(dict.fromkeys(cited_doc_ids)),
                cited_chunk_ids=cited_ids,
                candidate_evidence_chunks=[],
                candidate_evidence_chunk_ids=[],
                verification_label=ClaimVerificationLabel.UNVERIFIED,
                support_label="skipped",
                support_reason=structured_claim.skip_reason or "claim skipped before verification",
                verifier_method="claim_extractor_skip",
                verifier_score=0.0,
                raw_support_score=0.0,
                calibrated_confidence=None,
                confidence_status="unavailable",
                calibration_status="uncalibrated",
                evidence_reason=structured_claim.skip_reason or "claim skipped before verification",
                supporting_candidate_ids=[],
                contradicting_candidate_ids=[],
                neutral_candidate_ids=[],
                best_candidate_id=None,
                best_supporting_doc_id=None,
                evidence_mode="no_support",
                support_source_type="no_support",
                verifier_limitations=["Claim was not grounded because extraction marked it as non-verifiable."],
                external_signal_records=[],
                fallback_used=False,
                provenance={"analyzer": "ClaimEvidenceBuilder_v1"},
            )
        
        # Default claim-level verification
        output = self._verifier.verify(
            claim_text,
            query,
            candidates,
            metadata={
                "claim_id": claim_id,
                "source_sentence": (
                    structured_claim.source_sentence if structured_claim is not None else claim_text
                ),
                "cited_doc_ids": list(dict.fromkeys(cited_doc_ids)),
                "cited_chunk_ids": cited_ids,
                "claim_type": (
                    structured_claim.claim_type if structured_claim is not None else detect_claim_type(claim_text)
                ),
                "numbers": list(structured_claim.numbers) if structured_claim is not None else [],
                "critical_values": list(structured_claim.numbers) if structured_claim is not None else [],
                "dates": list(structured_claim.dates) if structured_claim is not None else [],
                "critical_dates": list(structured_claim.dates) if structured_claim is not None else [],
                "entities": list(structured_claim.entities) if structured_claim is not None else [],
                "critical_entities": list(structured_claim.entities) if structured_claim is not None else [],
                "atomicity_status": (
                    structured_claim.atomicity_status if structured_claim is not None else detect_atomicity(claim_text)
                ),
            },
        )
        
        triplets: list[ClaimTriplet] | None = None
        
        # Triplet verification path (optional)
        if self._triplet_extractor is not None and self._triplet_verifier is not None:
            try:
                triplets = self._triplet_extractor.extract(claim_text, source_claim_id=claim_id)
                if triplets:
                    triplet_results = self._triplet_verifier.verify_triplets(triplets, candidates)
                    output = self._aggregate_triplet_results(triplet_results, output)
            except Exception as exc:
                logger.warning("Triplet analysis failed for claim '%s': %s", claim_text[:60], exc)
                # Fall back to claim-level output already computed
        elif self._triplet_extractor is not None:
            # Extraction only path
            try:
                triplets = self._triplet_extractor.extract(claim_text, source_claim_id=claim_id)
            except Exception as exc:
                logger.warning("Triplet extraction failed for claim '%s': %s", claim_text[:60], exc)
                triplets = []
            
        # Apply calibration if available
        calib_features = {
            "raw_score": output.raw_score,
            "label": output.label,
            "claim_type": structured_claim.claim_type if structured_claim is not None else detect_claim_type(claim_text),
            "candidate_count": len(candidates),
            "value_match_count": len(output.value_matches),
            "value_conflict_count": len(output.value_conflicts),
            "fallback_used": output.fallback_used,
            "verifier_mode": output.verifier_name,
        }
        calibrated = self._calibrator.calibrate(calib_features)
        confidence_status = (
            "calibrated"
            if calibrated.confidence is not None and calibrated.status == "calibrated"
            else output.confidence_status
        )
        supporting_ids = list(dict.fromkeys(output.supporting_chunk_ids))
        best_supporting_doc_id = None
        if supporting_ids and supporting_ids[0] in chunks_by_id:
            best_supporting_doc_id = chunks_by_id[supporting_ids[0]].source_doc_id
        support_source_type = "no_support"
        if supporting_ids:
            if set(supporting_ids) & set(cited_ids):
                support_source_type = "exact_cited_chunk"
            elif best_supporting_doc_id is not None and best_supporting_doc_id in cited_doc_ids:
                support_source_type = "cited_doc_other_chunk"
            else:
                support_source_type = "retrieved_uncited_chunk"
        
        return ClaimEvidenceRecord(
            claim_id=claim_id,
            claim_text=claim_text,
            source_sentence=structured_claim.source_sentence if structured_claim is not None else claim_text,
            source_answer_span=(
                (structured_claim.source_start_char, structured_claim.source_end_char)
                if structured_claim is not None
                else None
            ),
            claim_type=structured_claim.claim_type if structured_claim is not None else detect_claim_type(claim_text),
            atomicity_status=structured_claim.atomicity_status if structured_claim is not None else detect_atomicity(claim_text),
            entities=list(structured_claim.entities) if structured_claim is not None else [],
            dates=list(structured_claim.dates) if structured_claim is not None else [],
            numbers=list(structured_claim.numbers) if structured_claim is not None else [],
            extraction_method=structured_claim.extraction_method if structured_claim is not None else "legacy_string_claim",
            extraction_reason=structured_claim.extraction_reason if structured_claim is not None else "legacy_string_claim",
            extraction_confidence=structured_claim.extraction_confidence if structured_claim is not None else None,
            extraction_warnings=list(structured_claim.extraction_warnings) if structured_claim is not None else [],
            skip_reason=structured_claim.skip_reason if structured_claim is not None else None,
            extracted_values=extract_value_mentions(claim_text),
            candidate_evidence_chunks=candidates,
            candidate_evidence_chunk_ids=[c.chunk_id for c in candidates],
            cited_doc_ids=list(dict.fromkeys(cited_doc_ids)),
            cited_chunk_ids=cited_ids,
            supporting_chunk_ids=output.supporting_chunk_ids,
            contradicting_chunk_ids=output.contradicting_chunk_ids,
            supporting_candidate_ids=output.supporting_chunk_ids,
            contradicting_candidate_ids=output.contradicting_chunk_ids,
            neutral_candidate_ids=output.neutral_chunk_ids,
            best_candidate_id=output.best_candidate_id,
            best_supporting_doc_id=best_supporting_doc_id,
            evidence_mode=output.evidence_mode,
            support_label=output.support_label,
            support_reason=output.rationale,
            raw_support_score=output.raw_score,
            support_source_type=support_source_type,
            verification_label=output.label,
            verifier_method=output.verifier_name,
            verifier_score=output.raw_score,
            label_reason=output.label_reason,
            calibrated_confidence=calibrated.confidence,
            confidence_status=confidence_status,
            calibration_status=calibrated.status,
            evidence_reason=output.rationale + (f" [Calibrated via {calibrated.source}]" if calibrated.source else ""),
            verifier_limitations=list(output.verifier_limitations),
            verifier_warnings=list(output.verifier_warnings),
            raw_entailment_response=output.raw_entailment_response,
            fallback_from=output.fallback_from,
            fallback_to=output.fallback_to,
            value_matches=output.value_matches,
            value_conflicts=output.value_conflicts,
            external_signal_records=list(output.external_signal_records),
            fallback_used=output.fallback_used,
            verifier_policy=output.verifier_policy,
            verifier_disagreement=output.verifier_disagreement,
            safety_gate_triggered=output.safety_gate_triggered,
            safety_gate_reason=output.safety_gate_reason,
            safety_gate_category=output.safety_gate_category,
            critical_fact_check_summary=output.critical_fact_check_summary,
            llm_label=output.llm_label,
            heuristic_label=output.heuristic_label,
            deterministic_gate_labels=output.deterministic_gate_labels,
            normalized_values_checked=output.normalized_values_checked,
            normalized_dates_checked=output.normalized_dates_checked,
            normalized_units_checked=output.normalized_units_checked,
            normalized_entities_checked=output.normalized_entities_checked,
            provenance={"analyzer": "ClaimEvidenceBuilder_v1"},
            structured_representation=StructuredClaimRepresentation(triplets=triplets) if triplets else None
        )

    def _aggregate_triplet_results(
        self,
        triplet_results: list[TripletVerificationResult],
        base_result: VerificationResult,
    ) -> VerificationResult:
        """
        Aggregate fine-grained triplet labels into a single claim label.
        
        Priority:
        1. If any triplet is contradicted -> claim is contradicted.
        2. If any triplet is abstain -> claim is abstain (or fallback).
        3. If any triplet is unsupported -> claim is unsupported.
        4. If all triplets are entailed -> claim is entailed.
        """
        if not triplet_results:
            return base_result
            
        labels = [r.label for r in triplet_results]
        
        if "contradicted" in labels:
            label = "contradicted"
        elif "abstain" in labels:
            label = "abstain"
        elif "unsupported" in labels:
            label = "unsupported"
        else:
            label = "entailed"
            
        # Select representative chunks
        supporting_ids = []
        contradicting_ids = []
        rationale_parts = []
        for res in triplet_results:
            if res.label == "entailed" and res.supporting_chunk_id:
                supporting_ids.append(res.supporting_chunk_id)
            if res.label == "contradicted" and res.contradicting_chunk_id:
                contradicting_ids.append(res.contradicting_chunk_id)
            if res.rationale:
                rationale_parts.append(f"Triplet {res.triplet_id or ''}: {res.rationale}")
                
        return VerificationResult(
            label=label,
            support_label=(
                "supported"
                if label == "entailed"
                else "contradicted"
                if label == "contradicted"
                else "unverifiable"
                if label == "abstain"
                else "insufficient_evidence"
            ),
            raw_score=sum(r.raw_score for r in triplet_results) / len(triplet_results),
            evidence_chunk_id=supporting_ids[0] if supporting_ids else (contradicting_ids[0] if contradicting_ids else None),
            evidence_span=None,
            rationale="; ".join(rationale_parts),
            verifier_name=triplet_results[0].method if triplet_results else "triplet_aggregator",
            supporting_chunk_ids=list(set(supporting_ids)),
            contradicting_chunk_ids=list(set(contradicting_ids)),
            best_candidate_id=supporting_ids[0] if supporting_ids else (contradicting_ids[0] if contradicting_ids else None),
            evidence_mode=(
                "multi_chunk"
                if len(set(supporting_ids)) > 1
                else "single_chunk"
                if supporting_ids
                else "no_support"
            ),
            aggregate_support_score=sum(r.raw_score for r in triplet_results) / len(triplet_results),
            aggregate_contradiction_score=max(
                (r.raw_score for r in triplet_results if r.label == "contradicted"),
                default=0.0,
            ),
            confidence_status="uncalibrated_heuristic_proxy",
            triplet_results=triplet_results,
        )
