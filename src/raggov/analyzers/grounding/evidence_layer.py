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
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk


logger = logging.getLogger(__name__)

NEGATION_SIGNALS = {"not", "never", "no", "no longer", "contrary to"}
ANCHOR_WEIGHT = 0.6
CONTENT_TERM_NORMALIZATIONS: dict[str, str] = {
    "grew": "increase",
    "grow": "increase",
    "growing": "increase",
    "increased": "increase",
    "increasing": "increase",
    "increases": "increase",
    "rose": "increase",
    "rising": "increase",
    "declined": "decrease",
    "decline": "decrease",
    "decreasing": "decrease",
    "decreased": "decrease",
    "falls": "decrease",
    "fell": "decrease",
    "annually": "annual",
    "yearly": "annual",
    "yoy": "annual",
    "yearoveryear": "annual",
}

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
_FINITE_VERBS = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|shall|must|should|can|may|"
    r"covers?|applies?|requires?|states?|provides?|allows?|prohibits?|entitles?)\b",
    re.IGNORECASE,
)


@dataclass
class EvidenceCandidate:
    """A retrieved chunk considered as evidence for a claim."""

    chunk_id: str
    raw_support_score: float
    is_best: bool = False


@dataclass
class VerificationOutput:
    """Raw output from a verifier before it is wrapped in a ClaimEvidenceRecord."""

    label: Literal["entailed", "unsupported", "contradicted"]
    verification_method: str
    raw_support_score: float
    evidence_reason: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    candidate_chunk_ids: list[str] = field(default_factory=list)
    contradicting_chunk_ids: list[str] = field(default_factory=list)
    value_matches: list[dict[str, str]] = field(default_factory=list)
    value_conflicts: list[dict[str, str]] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class ClaimEvidenceRecord:
    """
    Structured evidence record for a single claim.

    raw_support_score holds the heuristic overlap/anchor score.
    calibrated_confidence is always None for heuristic verifiers — no
    calibration has been performed. Do not treat raw_support_score as a
    calibrated confidence value.
    """

    claim_id: str
    claim_text: str
    claim_type: str
    atomicity_status: str
    extracted_values: list[ValueMention]
    candidate_evidence_chunks: list[EvidenceCandidate]
    supporting_chunk_ids: list[str]
    contradicting_chunk_ids: list[str]
    verification_label: Literal["entailed", "unsupported", "contradicted"]
    verification_method: str
    raw_support_score: float
    calibrated_confidence: float | None
    calibration_status: str
    evidence_reason: str
    value_matches: list[dict[str, str]]
    value_conflicts: list[dict[str, str]]
    fallback_used: bool


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


class HeuristicValueOverlapVerifier:
    """
    Heuristic v0 evidence verifier using term overlap, anchor matching, and
    value alignment. This is NOT an NLI verifier, RefChecker implementation,
    RAGChecker implementation, or calibrated confidence system.

    raw_support_score produced by this verifier is an uncalibrated heuristic
    score, not a calibrated confidence value.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def verify(self, claim: str, query: str, chunks: list[RetrievedChunk]) -> VerificationOutput:
        """
        Run structured verification with automatic fallback to deterministic.

        Never raises to the caller. If forced error is configured or structured
        verification fails, falls back to deterministic and sets fallback_used=True.
        """
        try:
            if bool(self._config.get("force_structured_verifier_error", False)):
                raise RuntimeError("forced structured verifier failure")
            return self._verify_structured(claim, _query=query, chunks=chunks)
        except Exception as exc:
            logger.warning(
                "Structured claim verifier failed, falling back to deterministic: %s", exc
            )
            base = self._verify_deterministic(claim, chunks)
            return VerificationOutput(
                label=base.label,
                verification_method=base.verification_method,
                raw_support_score=base.raw_support_score,
                evidence_reason=(
                    "Structured verifier unavailable; fell back to deterministic overlap/anchor check"
                ),
                supporting_chunk_ids=base.supporting_chunk_ids,
                candidate_chunk_ids=base.candidate_chunk_ids,
                contradicting_chunk_ids=base.contradicting_chunk_ids,
                value_matches=base.value_matches,
                value_conflicts=base.value_conflicts,
                fallback_used=True,
            )

    def _verify_structured(
        self, claim: str, _query: str, chunks: list[RetrievedChunk]
    ) -> VerificationOutput:
        """Structured evidence-aware verification using best-evidence chunk analysis."""
        if not chunks:
            return VerificationOutput(
                label="unsupported",
                verification_method="structured_claim_verifier_v1",
                raw_support_score=0.0,
                evidence_reason="No retrieved chunks available for evidence matching.",
            )

        claim_terms = self._content_terms(claim)
        claim_anchors = self._extract_anchors(claim)
        if not claim_terms:
            return VerificationOutput(
                label="unsupported",
                verification_method="structured_claim_verifier_v1",
                raw_support_score=0.0,
                evidence_reason="Claim has no meaningful content terms to verify.",
            )

        best_chunk, best_score = self._best_evidence_chunk(claim_terms, claim_anchors, chunks)
        if best_chunk is None:
            return VerificationOutput(
                label="unsupported",
                verification_method="structured_claim_verifier_v1",
                raw_support_score=0.0,
                evidence_reason="No candidate evidence chunk matched claim content terms.",
            )

        claim_has_numeric_anchor = any(ch.isdigit() for anchor in claim_anchors for ch in anchor)
        chunk_anchor_set = set(self._extract_anchors(best_chunk.text))
        claim_anchor_set = set(claim_anchors)
        anchor_hits = len(claim_anchor_set & chunk_anchor_set) if claim_anchor_set else 0
        numeric_anchor_mismatch = (
            bool(claim_anchor_set) and claim_has_numeric_anchor and anchor_hits == 0
        )

        if self._contains_negation_of_terms(best_chunk.text, claim_terms):
            return VerificationOutput(
                label="contradicted",
                verification_method="value_aware_structured_claim_verifier_v1",
                raw_support_score=best_score,
                evidence_reason=(
                    "Best evidence chunk contains negation near claim terms, "
                    "indicating contradiction."
                ),
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
            )

        value_matches, value_conflicts, missing_critical_value = find_value_alignment(
            claim, best_chunk.text
        )
        if value_conflicts:
            conflict = value_conflicts[0]
            return VerificationOutput(
                label="contradicted",
                verification_method="value_aware_structured_claim_verifier_v1",
                raw_support_score=best_score,
                evidence_reason=(
                    f"Claim states {conflict['claim_value']}, but evidence states "
                    f"{conflict['evidence_value']} for the same {conflict['value_type']} context."
                ),
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        if missing_critical_value:
            snippet = _first_claim_value_snippet(claim)
            return VerificationOutput(
                label="unsupported",
                verification_method="value_aware_structured_claim_verifier_v1",
                raw_support_score=best_score,
                evidence_reason=(
                    f"Claim value {snippet} is not present in related evidence."
                    if snippet
                    else "Claim contains critical value details not present in related evidence."
                ),
                candidate_chunk_ids=[best_chunk.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        if numeric_anchor_mismatch and not value_matches:
            return VerificationOutput(
                label="unsupported",
                verification_method="value_aware_structured_claim_verifier_v1",
                raw_support_score=best_score,
                evidence_reason=(
                    "High lexical overlap but factual anchors "
                    "(numbers/dates/entities) do not match."
                ),
                candidate_chunk_ids=[best_chunk.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        label: Literal["entailed", "unsupported", "contradicted"] = (
            "entailed"
            if (best_score >= 0.5 or (value_matches and best_score >= 0.2))
            else "unsupported"
        )
        supporting = [best_chunk.chunk_id] if (label == "entailed" and best_score >= 0.2) else []
        reason = (
            "Claim supported by best evidence chunk via content-term and anchor match."
            if label == "entailed"
            else "Best evidence chunk does not provide enough content-term/anchor support."
        )
        return VerificationOutput(
            label=label,
            verification_method="value_aware_structured_claim_verifier_v1",
            raw_support_score=best_score,
            evidence_reason=reason,
            supporting_chunk_ids=supporting,
            candidate_chunk_ids=[best_chunk.chunk_id],
            value_matches=value_matches,
            value_conflicts=value_conflicts,
        )

    def _verify_deterministic(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> VerificationOutput:
        """Score grounding via normalized content-term coverage plus factual anchors."""
        claim_terms = self._content_terms(claim)
        claim_anchors = self._extract_anchors(claim)
        if not claim_terms:
            return VerificationOutput(
                label="unsupported",
                verification_method="deterministic_overlap_anchor_v0",
                raw_support_score=0.0,
                evidence_reason="Claim has no meaningful content terms after normalization.",
            )

        best_chunk: RetrievedChunk | None = None
        best_score = 0.0
        term_weight = 1.0 - float(self._config.get("anchor_weight", ANCHOR_WEIGHT))
        anchor_weight = float(self._config.get("anchor_weight", ANCHOR_WEIGHT))

        for chunk in chunks:
            chunk_terms = self._content_terms(chunk.text)
            term_coverage = len(claim_terms & chunk_terms) / len(claim_terms)
            if claim_anchors:
                chunk_anchors = set(self._extract_anchors(chunk.text))
                anchor_hits = len(set(claim_anchors) & chunk_anchors)
                anchor_coverage = anchor_hits / len(set(claim_anchors))
                score = (term_weight * term_coverage) + (anchor_weight * anchor_coverage)
            else:
                score = term_coverage
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is not None and self._contains_negation_of_terms(
            best_chunk.text, claim_terms
        ):
            return VerificationOutput(
                label="contradicted",
                verification_method="deterministic_overlap_anchor_v0",
                raw_support_score=best_score,
                evidence_reason="Negation detected near overlapping claim terms.",
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
            )

        value_matches: list[dict[str, str]] = []
        value_conflicts: list[dict[str, str]] = []
        if best_chunk is not None:
            value_matches, value_conflicts, missing_critical = find_value_alignment(
                claim, best_chunk.text
            )
            if value_conflicts:
                conflict = value_conflicts[0]
                return VerificationOutput(
                    label="contradicted",
                    verification_method="deterministic_overlap_anchor_v0",
                    raw_support_score=best_score,
                    evidence_reason=(
                        f"Claim states {conflict['claim_value']}, but evidence states "
                        f"{conflict['evidence_value']} for the same {conflict['value_type']} context."
                    ),
                    candidate_chunk_ids=[best_chunk.chunk_id],
                    contradicting_chunk_ids=[best_chunk.chunk_id],
                    value_matches=value_matches,
                    value_conflicts=value_conflicts,
                )
            if missing_critical:
                snippet = _first_claim_value_snippet(claim)
                return VerificationOutput(
                    label="unsupported",
                    verification_method="deterministic_overlap_anchor_v0",
                    raw_support_score=best_score,
                    evidence_reason=(
                        f"Claim value {snippet} is not present in related evidence."
                        if snippet
                        else "Claim contains critical value details not present in related evidence."
                    ),
                    candidate_chunk_ids=[best_chunk.chunk_id],
                    value_matches=value_matches,
                    value_conflicts=value_conflicts,
                )

        label: Literal["entailed", "unsupported", "contradicted"] = (
            "entailed"
            if (best_score >= 0.5 or (value_matches and best_score >= 0.2))
            else "unsupported"
        )
        supporting = (
            [best_chunk.chunk_id]
            if (label == "entailed" and best_score >= 0.2 and best_chunk is not None)
            else []
        )
        candidate = [best_chunk.chunk_id] if best_chunk is not None else []
        reason = (
            "Best chunk crosses support threshold via overlap and anchor scoring."
            if label == "entailed"
            else "Best chunk score below support threshold."
        )
        return VerificationOutput(
            label=label,
            verification_method="deterministic_overlap_anchor_v0",
            raw_support_score=best_score,
            evidence_reason=reason,
            supporting_chunk_ids=supporting,
            candidate_chunk_ids=candidate,
            value_matches=value_matches,
            value_conflicts=value_conflicts,
        )

    def _best_evidence_chunk(
        self,
        claim_terms: set[str],
        claim_anchors: list[str],
        chunks: list[RetrievedChunk],
    ) -> tuple[RetrievedChunk | None, float]:
        best_chunk: RetrievedChunk | None = None
        best_score = 0.0
        term_weight = 1.0 - float(self._config.get("anchor_weight", ANCHOR_WEIGHT))
        anchor_weight = float(self._config.get("anchor_weight", ANCHOR_WEIGHT))
        for chunk in chunks:
            chunk_terms = self._content_terms(chunk.text)
            term_coverage = len(claim_terms & chunk_terms) / len(claim_terms)
            if claim_anchors:
                chunk_anchors = set(self._extract_anchors(chunk.text))
                anchor_hits = len(set(claim_anchors) & chunk_anchors)
                anchor_coverage = anchor_hits / len(set(claim_anchors))
                score = (term_weight * term_coverage) + (anchor_weight * anchor_coverage)
            else:
                score = term_coverage
            if score > best_score:
                best_score = score
                best_chunk = chunk
        return best_chunk, best_score

    def _contains_negation_of_terms(self, text: str, terms: set[str]) -> bool:
        tokens = self._tokens(text)
        for index, token in enumerate(tokens):
            normalized = self._normalize_content_term(token)
            window = tokens[max(0, index - 5) : index + 6]
            window_text = " ".join(window)
            if normalized in terms and any(
                re.search(rf"\b{re.escape(signal)}\b", window_text)
                for signal in NEGATION_SIGNALS
            ):
                return True
        return False

    def _content_terms(self, text: str) -> set[str]:
        content_terms: set[str] = set()
        for token in self._tokens(text):
            if token in STOPWORDS:
                continue
            normalized = self._normalize_content_term(token)
            if not normalized:
                continue
            if normalized.isdigit() or len(normalized) > 2:
                content_terms.add(normalized)
        return content_terms

    def _normalize_content_term(self, token: str) -> str:
        if not token:
            return ""
        normalized = CONTENT_TERM_NORMALIZATIONS.get(token, token)
        if normalized.endswith("ies") and len(normalized) > 4:
            return normalized[:-3] + "y"
        if normalized.endswith("s") and len(normalized) > 4 and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized

    def _extract_anchors(self, text: str) -> list[str]:
        anchors: list[str] = []
        lowered = text.lower()
        anchors.extend(
            m.group(0) for m in re.finditer(r"(?:[$€£])?\d[\d,]*(?:\.\d+)?%?", lowered)
        )
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            value = m.group(0)
            if " " in value or m.start() > 0:
                anchors.append(value.lower())
        for m in re.finditer(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b", text):
            anchors.append(m.group(0).lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            if anchor in seen:
                continue
            seen.add(anchor)
            deduped.append(anchor)
        return deduped

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


class ClaimEvidenceBuilder:
    """
    Builds ClaimEvidenceRecord objects by coordinating ClaimExtractor and
    HeuristicValueOverlapVerifier. Entry point for the heuristic evidence layer.
    """

    def __init__(self, verifier: HeuristicValueOverlapVerifier) -> None:
        self._verifier = verifier

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
        output = self._verifier.verify(claim, query, chunks)
        return ClaimEvidenceRecord(
            claim_id=f"claim_{index:03d}",
            claim_text=claim,
            claim_type=detect_claim_type(claim),
            atomicity_status=detect_atomicity(claim),
            extracted_values=extract_value_mentions(claim),
            candidate_evidence_chunks=[
                EvidenceCandidate(
                    chunk_id=cid,
                    raw_support_score=output.raw_support_score,
                    is_best=True,
                )
                for cid in output.candidate_chunk_ids
            ],
            supporting_chunk_ids=output.supporting_chunk_ids,
            contradicting_chunk_ids=output.contradicting_chunk_ids,
            verification_label=output.label,
            verification_method=output.verification_method,
            raw_support_score=output.raw_support_score,
            calibrated_confidence=None,
            calibration_status="uncalibrated",
            evidence_reason=output.evidence_reason,
            value_matches=output.value_matches,
            value_conflicts=output.value_conflicts,
            fallback_used=output.fallback_used,
        )


def _first_claim_value_snippet(claim_text: str) -> str | None:
    match = re.search(
        r"(?:[$₹€£]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?%?\b"
        r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))*\b)",
        claim_text.lower(),
    )
    return match.group(0) if match else None
