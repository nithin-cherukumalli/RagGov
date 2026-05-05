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
from raggov.analyzers.grounding.verifiers import EvidenceVerifier, VerificationResult
from raggov.analyzers.grounding.triplets import ClaimTriplet, TripletExtractor
from raggov.calibration.claim_calibration import ClaimCalibrationModel, CalibratedClaimConfidence, CalibrationMode


logger = logging.getLogger(__name__)

NEGATION_SIGNALS = {"not", "never", "no", "no longer", "contrary to"}

_GO_PATTERN = re.compile(r"g\.o", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%|\bpercent\b|\bamount\b|\bthreshold\b|\bceiling\b|\blimit\b"
    r"|[$₹€£]|\brupees?\b|\brs\.?\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b|\bdeadline\b|\beffective\s+date\b",
    re.IGNORECASE,
)
_ELIGIBILITY_PATTERN = re.compile(
    r"\beligib\w*|\bqualif\w*|\bentitl\w*|\bwho\s+can\b|\bwho\s+is\b",
    re.IGNORECASE,
)
_DEFINITION_PATTERN = re.compile(
    r"\bmeans?\b|\brefers?\b|\bis\s+defined\b|\bdenotes?\b",
    re.IGNORECASE,
)
_POLICY_PATTERN = re.compile(
    r"\brule\b|\bpolicy\b|\bshall\b|\bmust\b|\brequired\b|\bmandatory\b"
    r"|\bprohibit\b|\bpermit\b",
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

    Priority: go_number > numeric > date_or_deadline > definition >
    eligibility > policy_rule > general_factual
    """
    if _GO_PATTERN.search(claim):
        return "go_number"
    if _NUMERIC_PATTERN.search(claim):
        return "numeric"
    if _DATE_PATTERN.search(claim):
        return "date_or_deadline"
    if _DEFINITION_PATTERN.search(claim):
        return "definition"
    if _ELIGIBILITY_PATTERN.search(claim):
        return "eligibility"
    if _POLICY_PATTERN.search(claim):
        return "policy_rule"
    return "general_factual"


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
        claims: list[str],
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[ClaimEvidenceRecord]:
        return [
            self._build_single(claim, index, query, chunks)
            for index, claim in enumerate(claims, start=1)
        ]

    def _build_single(
        self,
        claim: str,
        index: int,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> ClaimEvidenceRecord:
        candidates = self._selector.select_candidates(claim, query, chunks)
        claim_id = f"claim_{index:03d}"
        
        # Default claim-level verification
        output = self._verifier.verify(
            claim,
            query,
            candidates,
            metadata={"claim_id": claim_id},
        )
        
        triplets: list[ClaimTriplet] | None = None
        
        # Triplet verification path (optional)
        if self._triplet_extractor is not None and self._triplet_verifier is not None:
            try:
                triplets = self._triplet_extractor.extract(claim, source_claim_id=claim_id)
                if triplets:
                    triplet_results = self._triplet_verifier.verify_triplets(triplets, candidates)
                    output = self._aggregate_triplet_results(triplet_results, output)
            except Exception as exc:
                logger.warning("Triplet analysis failed for claim '%s': %s", claim[:60], exc)
                # Fall back to claim-level output already computed
        elif self._triplet_extractor is not None:
            # Extraction only path
            try:
                triplets = self._triplet_extractor.extract(claim, source_claim_id=claim_id)
            except Exception as exc:
                logger.warning("Triplet extraction failed for claim '%s': %s", claim[:60], exc)
                triplets = []

        # Parse citations from claim text
        cited_ids = []
        for match in _CITATION_REGEX.finditer(claim):
            cited_ids.append(f"c{match.group(1)}")
            
        # Apply calibration if available
        calib_features = {
            "raw_score": output.raw_score,
            "label": output.label,
            "claim_type": detect_claim_type(claim),
            "candidate_count": len(candidates),
            "value_match_count": len(output.value_matches),
            "value_conflict_count": len(output.value_conflicts),
            "fallback_used": output.fallback_used,
            "verifier_mode": output.verifier_name,
        }
        calibrated = self._calibrator.calibrate(calib_features)
        
        return ClaimEvidenceRecord(
            claim_id=claim_id,
            claim_text=claim,
            claim_type=detect_claim_type(claim),
            atomicity_status=detect_atomicity(claim),
            extracted_values=extract_value_mentions(claim),
            candidate_evidence_chunks=candidates,
            candidate_evidence_chunk_ids=[c.chunk_id for c in candidates],
            cited_chunk_ids=cited_ids,
            supporting_chunk_ids=output.supporting_chunk_ids,
            contradicting_chunk_ids=output.contradicting_chunk_ids,
            verification_label=output.label,
            verifier_method=output.verifier_name,
            verifier_score=output.raw_score,
            calibrated_confidence=calibrated.confidence,
            calibration_status=calibrated.status,
            evidence_reason=output.rationale + (f" [Calibrated via {calibrated.source}]" if calibrated.source else ""),
            value_matches=output.value_matches,
            value_conflicts=output.value_conflicts,
            external_signal_records=list(output.external_signal_records),
            fallback_used=output.fallback_used,
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
            raw_score=sum(r.raw_score for r in triplet_results) / len(triplet_results),
            evidence_chunk_id=supporting_ids[0] if supporting_ids else (contradicting_ids[0] if contradicting_ids else None),
            evidence_span=None,
            rationale="; ".join(rationale_parts),
            verifier_name=triplet_results[0].method if triplet_results else "triplet_aggregator",
            supporting_chunk_ids=list(set(supporting_ids)),
            contradicting_chunk_ids=list(set(contradicting_ids)),
            triplet_results=triplet_results,
        )

