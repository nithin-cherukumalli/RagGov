"""
Verification Layer for GovRAG grounding analysis.

Defines the EvidenceVerifier interface and provides heuristic, LLM, and abstaining implementations.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.value_extraction import find_value_alignment

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Standardized output from any claim evidence verifier."""

    label: Literal["entailed", "unsupported", "contradicted", "abstain"]
    raw_score: float
    evidence_chunk_id: str | None
    evidence_span: str | None
    rationale: str
    verifier_name: str
    fallback_used: bool = False
    error_info: str | None = None
    
    # Internal fields for compatibility with ClaimEvidenceRecord builders
    supporting_chunk_ids: list[str] = field(default_factory=list)
    candidate_chunk_ids: list[str] = field(default_factory=list)
    contradicting_chunk_ids: list[str] = field(default_factory=list)
    value_matches: list[dict[str, str]] = field(default_factory=list)
    value_conflicts: list[dict[str, str]] = field(default_factory=list)
    external_signal_records: list[dict[str, Any]] = field(default_factory=list)
    triplet_results: list[TripletVerificationResult] = field(default_factory=list)


@dataclass
class TripletVerificationResult:
    """Standardized output from a triplet-level verifier."""

    label: Literal["entailed", "unsupported", "contradicted", "abstain"]
    raw_score: float
    supporting_chunk_id: str | None = None
    contradicting_chunk_id: str | None = None
    rationale: str = ""
    method: str = ""
    triplet_id: str | None = None


class EvidenceVerifier(ABC):
    """Abstract interface for checking if a claim is supported by candidate evidence."""

    @abstractmethod
    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        pass


class TripletVerifier(ABC):
    """Abstract interface for checking if claim-triplets are supported by evidence."""

    @abstractmethod
    def verify_triplets(
        self,
        triplets: list[Any],  # list[ClaimTriplet]
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> list[TripletVerificationResult]:
        pass


class AbstainingVerifier(EvidenceVerifier):
    """A fallback verifier that always abstains from judgment."""

    def __init__(self, reason: str = "Verifier abstained or failed.", error_info: str | None = None):
        self.reason = reason
        self.error_info = error_info

    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            label="abstain",
            raw_score=0.0,
            evidence_chunk_id=None,
            evidence_span=None,
            rationale=self.reason,
            verifier_name="abstaining_verifier",
            fallback_used=True,
            error_info=self.error_info,
            candidate_chunk_ids=[c.chunk_id for c in candidates],
        )


class StructuredLLMClaimVerifier(EvidenceVerifier):
    """Verifies claims using an LLM requested to output structured JSON reasoning."""

    def __init__(self, config: dict[str, Any]):
        self._client = config.get("llm_client")
        if not self._client:
            raise ValueError("StructuredLLMClaimVerifier requires an 'llm_client' in config.")
            
    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(
                label="unsupported",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No candidate evidence chunks available.",
                verifier_name="structured_llm_claim_verifier_v1",
            )
            
        try:
            return self._call_llm(claim, candidates)
        except Exception as exc:
            logger.warning("LLM claim grounding failed, returning Abstaining: %s", exc)
            return AbstainingVerifier(
                reason=f"LLM verifier failed: {exc}",
                error_info=str(exc)
            ).verify(claim, query, candidates, metadata)

    def _call_llm(self, claim: str, candidates: list[EvidenceCandidate]) -> VerificationResult:
        prompt = self._prompt(claim, candidates)
        if hasattr(self._client, "chat"):
            response = self._client.chat(prompt)
        elif hasattr(self._client, "complete"):
            response = self._client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
            
        parsed = self._parse_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM grounding response must be a JSON object")
            
        label = parsed.get("label")
        if label not in {"entailed", "unsupported", "contradicted", "abstain"}:
            raise ValueError(f"invalid LLM grounding label: {label}")
            
        evidence_chunk_id = parsed.get("evidence_chunk_id")
        supporting_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id and label == "entailed" else []
        contradicting_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id and label == "contradicted" else []
        candidate_ids = [c.chunk_id for c in candidates]
        
        return VerificationResult(
            label=label,
            raw_score=float(parsed.get("confidence", 0.0)),
            evidence_chunk_id=str(evidence_chunk_id) if evidence_chunk_id else None,
            evidence_span=parsed.get("evidence_span"),
            rationale=str(parsed.get("rationale", "LLM verifier output")),
            verifier_name="structured_llm_claim_verifier_v1",
            supporting_chunk_ids=supporting_chunk_ids,
            candidate_chunk_ids=candidate_ids,
            contradicting_chunk_ids=contradicting_chunk_ids,
        )

    def _prompt(self, claim: str, candidates: list[EvidenceCandidate]) -> str:
        relevant_chunks = "\n\n".join(
            f"[{c.chunk_id}] {c.chunk_text}" for c in candidates
        )
        return (
            "Does the evidence explicitly support, contradict, or not provide enough information for the claim?\n"
            "Rules:\n"
            "- 'unsupported' if evidence is related but does not contain enough information.\n"
            "- 'contradicted' only if evidence conflicts with the claim.\n"
            "- 'entailed' only if evidence directly supports the full claim.\n"
            "- 'abstain' if the prompt is malformed or fundamentally unanswerable.\n"
            "- Do not use outside knowledge.\n"
            "- Return strict JSON.\n\n"
            f"Context: {relevant_chunks}\n"
            f"Claim: {claim}\n"
            'Answer with JSON: {"label": "entailed"|"unsupported"|"contradicted"|"abstain", '
            '"confidence": 0.0-1.0, "evidence_chunk_id": "chunk_id or null", '
            '"evidence_span": "exact quote from evidence or null", '
            '"rationale": "short reason"}'
        )

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
            else:
                return response
        if not isinstance(response, str):
            response = str(response)
        return json.loads(response)


class LLMTripletVerifierV1(TripletVerifier):
    """
    Verifies claim-triplets against evidence using an LLM.
    
    RefChecker-inspired.
    """

    def __init__(self, config: dict[str, Any]):
        self._client = config.get("llm_client")
        if not self._client:
            raise ValueError("LLMTripletVerifierV1 requires an 'llm_client' in config.")
        self._method = "llm_triplet_verifier_v1"

    def verify_triplets(
        self,
        triplets: list[Any],
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> list[TripletVerificationResult]:
        if not triplets or not candidates:
            return []
            
        results: list[TripletVerificationResult] = []
        for triplet in triplets:
            # For each triplet, we check against the best candidate
            # RefChecker usually checks against all, but for v0 we use the best.
            best = candidates[0]
            try:
                res = self._verify_single(triplet, best)
                results.append(res)
            except Exception as exc:
                logger.warning("Triplet verification failed for %s: %s", triplet, exc)
                results.append(TripletVerificationResult(
                    label="abstain",
                    raw_score=0.0,
                    rationale=f"Error: {exc}",
                    method=self._method,
                    triplet_id=getattr(triplet, "triplet_id", None)
                ))
        return results

    def _verify_single(self, triplet: Any, candidate: EvidenceCandidate) -> TripletVerificationResult:
        prompt = self._prompt(triplet, candidate)
        if hasattr(self._client, "chat"):
            response = self._client.chat(prompt)
        elif hasattr(self._client, "complete"):
            response = self._client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
            
        parsed = self._parse_response(response)
        label = parsed.get("label", "unsupported")
        if label not in {"entailed", "unsupported", "contradicted", "abstain"}:
            label = "unsupported"
            
        return TripletVerificationResult(
            label=label,
            raw_score=float(parsed.get("confidence", 0.0)),
            supporting_chunk_id=candidate.chunk_id if label == "entailed" else None,
            contradicting_chunk_id=candidate.chunk_id if label == "contradicted" else None,
            rationale=str(parsed.get("rationale", "")),
            method=self._method,
            triplet_id=getattr(triplet, "triplet_id", None)
        )

    def _prompt(self, triplet: Any, candidate: EvidenceCandidate) -> str:
        triplet_str = f"({triplet.subject}, {triplet.predicate}, {triplet.object})"
        if triplet.values:
            triplet_str += f" with values: {triplet.values}"
        
        return (
            "Given a triplet and an evidence chunk, decide whether the evidence entails, "
            "contradicts, or does not provide enough information for the triplet.\n"
            "Use only provided evidence.\n\n"
            f"Evidence: {candidate.chunk_text}\n"
            f"Triplet: {triplet_str}\n\n"
            'Answer with JSON: {"label": "entailed"|"unsupported"|"contradicted"|"abstain", '
            '"confidence": 0.0-1.0, "rationale": "short reason"}'
        )

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
            else:
                return response
        if not isinstance(response, str):
            response = str(response)
        # Strip markdown if present
        response = re.sub(r"```(?:json)?", "", response).strip().strip("`")
        return json.loads(response)


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


class HeuristicValueOverlapVerifier(EvidenceVerifier):
    """
    Heuristic v0 evidence verifier using term overlap, anchor matching, and
    value alignment.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        # Configurable thresholds — defaults are the original hardcoded values.
        # Use sweep_thresholds.py to explore alternatives; do NOT change defaults
        # without running the full eval harness and reviewing false_pass_rate.
        self._support_threshold: float = float(
            config.get("support_threshold", 0.5)
        )
        self._value_match_score_boost: float = float(
            config.get("value_match_score_boost", 0.2)
        )
        # When a critical numeric/date value appears in the claim but not in evidence:
        # 'unsupported' (default, conservative) or 'contradicted' (aggressive)
        self._missing_critical_value_behavior: str = str(
            config.get("missing_critical_value_behavior", "unsupported")
        )

    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """
        Run structured verification with automatic fallback to deterministic.
        """
        try:
            if bool(self._config.get("force_structured_verifier_error", False)):
                raise RuntimeError("forced structured verifier failure")
            return self._verify_structured(claim, _query=query, candidates=candidates)
        except Exception as exc:
            logger.warning(
                "Structured claim verifier failed, falling back to deterministic: %s", exc
            )
            base = self._verify_deterministic(claim, candidates)
            base.fallback_used = True
            base.rationale = "Structured verifier unavailable; fell back to deterministic overlap/anchor check."
            return base

    def _verify_structured(
        self, claim: str, _query: str, candidates: list[EvidenceCandidate]
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(
                label="unsupported",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No retrieved chunks available for evidence matching.",
                verifier_name="value_aware_structured_claim_verifier_v1",
            )

        best_candidate = candidates[0]
        best_score = best_candidate.raw_support_score
        
        claim_terms = self._content_terms(claim)
        claim_anchors = self._extract_anchors(claim)
        if not claim_terms:
            return VerificationResult(
                label="unsupported",
                raw_score=0.0,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale="Claim has no meaningful content terms to verify.",
                verifier_name="value_aware_structured_claim_verifier_v1",
            )

        claim_has_numeric_anchor = any(ch.isdigit() for anchor in claim_anchors for ch in anchor)
        chunk_anchor_set = set(self._extract_anchors(best_candidate.chunk_text))
        claim_anchor_set = set(claim_anchors)
        anchor_hits = len(claim_anchor_set & chunk_anchor_set) if claim_anchor_set else 0
        numeric_anchor_mismatch = (
            bool(claim_anchor_set) and claim_has_numeric_anchor and anchor_hits == 0
        )

        if self._contains_negation_of_terms(best_candidate.chunk_text, claim_terms):
            return VerificationResult(
                label="contradicted",
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale="Best evidence chunk contains negation near claim terms, indicating contradiction.",
                verifier_name="value_aware_structured_claim_verifier_v1",
                candidate_chunk_ids=[best_candidate.chunk_id],
                contradicting_chunk_ids=[best_candidate.chunk_id],
            )

        value_matches, value_conflicts, missing_critical_value = find_value_alignment(
            claim, best_candidate.chunk_text
        )
        if value_conflicts:
            conflict = value_conflicts[0]
            return VerificationResult(
                label="contradicted",
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale=(
                    f"Claim states {conflict['claim_value']}, but evidence states "
                    f"{conflict['evidence_value']} for the same {conflict['value_type']} context."
                ),
                verifier_name="value_aware_structured_claim_verifier_v1",
                candidate_chunk_ids=[best_candidate.chunk_id],
                contradicting_chunk_ids=[best_candidate.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        if missing_critical_value:
            snippet = _first_claim_value_snippet(claim)
            missing_label: Literal["entailed", "unsupported", "contradicted"] = (
                "contradicted"
                if self._missing_critical_value_behavior == "contradicted"
                else "unsupported"
            )
            return VerificationResult(
                label=missing_label,
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale=(
                    f"Claim value {snippet} is not present in related evidence."
                    if snippet
                    else "Claim contains critical value details not present in related evidence."
                ),
                verifier_name="value_aware_structured_claim_verifier_v1",
                candidate_chunk_ids=[best_candidate.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        if numeric_anchor_mismatch and not value_matches:
            return VerificationResult(
                label="unsupported",
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale="High lexical overlap but factual anchors (numbers/dates/entities) do not match.",
                verifier_name="value_aware_structured_claim_verifier_v1",
                candidate_chunk_ids=[best_candidate.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        label: Literal["entailed", "unsupported", "contradicted"] = (
            "entailed"
            if (
                best_score >= self._support_threshold
                or (value_matches and best_score >= self._value_match_score_boost)
            )
            else "unsupported"
        )
        supporting = [
            best_candidate.chunk_id
        ] if (label == "entailed" and best_score >= self._value_match_score_boost) else []
        reason = (
            "Claim supported by best evidence chunk via content-term and anchor match."
            if label == "entailed"
            else "Best chunk score below support threshold."
        )
        return VerificationResult(
            label=label,
            raw_score=best_score,
            evidence_chunk_id=best_candidate.chunk_id,
            evidence_span=None,
            rationale=reason,
            verifier_name="value_aware_structured_claim_verifier_v1",
            supporting_chunk_ids=supporting,
            candidate_chunk_ids=[best_candidate.chunk_id],
            value_matches=value_matches,
            value_conflicts=value_conflicts,
        )

    def _verify_deterministic(
        self, claim: str, candidates: list[EvidenceCandidate]
    ) -> VerificationResult:
        claim_terms = self._content_terms(claim)
        if not claim_terms:
            return VerificationResult(
                label="unsupported",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="Claim has no meaningful content terms after normalization.",
                verifier_name="deterministic_overlap_anchor_v0",
            )

        if not candidates:
            return VerificationResult(
                label="unsupported",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No candidate evidence chunk matched claim content terms.",
                verifier_name="deterministic_overlap_anchor_v0",
            )

        best_candidate = candidates[0]
        best_score = best_candidate.raw_support_score

        if self._contains_negation_of_terms(best_candidate.chunk_text, claim_terms):
            return VerificationResult(
                label="contradicted",
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale="Negation detected near overlapping claim terms.",
                verifier_name="deterministic_overlap_anchor_v0",
                candidate_chunk_ids=[best_candidate.chunk_id],
                contradicting_chunk_ids=[best_candidate.chunk_id],
            )

        value_matches, value_conflicts, missing_critical = find_value_alignment(
            claim, best_candidate.chunk_text
        )
        if value_conflicts:
            conflict = value_conflicts[0]
            return VerificationResult(
                label="contradicted",
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale=(
                    f"Claim states {conflict['claim_value']}, but evidence states "
                    f"{conflict['evidence_value']} for the same {conflict['value_type']} context."
                ),
                verifier_name="deterministic_overlap_anchor_v0",
                candidate_chunk_ids=[best_candidate.chunk_id],
                contradicting_chunk_ids=[best_candidate.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )
        if missing_critical:
            snippet = _first_claim_value_snippet(claim)
            missing_label_det: Literal["entailed", "unsupported", "contradicted"] = (
                "contradicted"
                if self._missing_critical_value_behavior == "contradicted"
                else "unsupported"
            )
            return VerificationResult(
                label=missing_label_det,
                raw_score=best_score,
                evidence_chunk_id=best_candidate.chunk_id,
                evidence_span=None,
                rationale=(
                    f"Claim value {snippet} is not present in related evidence."
                    if snippet
                    else "Claim contains critical value details not present in related evidence."
                ),
                verifier_name="deterministic_overlap_anchor_v0",
                candidate_chunk_ids=[best_candidate.chunk_id],
                value_matches=value_matches,
                value_conflicts=value_conflicts,
            )

        label: Literal["entailed", "unsupported", "contradicted"] = (
            "entailed"
            if (
                best_score >= self._support_threshold
                or (value_matches and best_score >= self._value_match_score_boost)
            )
            else "unsupported"
        )
        supporting = (
            [best_candidate.chunk_id]
            if (label == "entailed" and best_score >= self._value_match_score_boost)
            else []
        )
        reason = (
            "Best chunk crosses support threshold via overlap and anchor scoring."
            if label == "entailed"
            else "Best chunk score below support threshold."
        )
        return VerificationResult(
            label=label,
            raw_score=best_score,
            evidence_chunk_id=best_candidate.chunk_id,
            evidence_span=None,
            rationale=reason,
            verifier_name="deterministic_overlap_anchor_v0",
            supporting_chunk_ids=supporting,
            candidate_chunk_ids=[best_candidate.chunk_id],
            value_matches=value_matches,
            value_conflicts=value_conflicts,
        )

    def _contains_negation_of_terms(self, text: str, terms: set[str]) -> bool:
        tokens = self._tokens(text)
        from raggov.analyzers.grounding.evidence_layer import NEGATION_SIGNALS
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
        from raggov.analyzers.retrieval.scope import STOPWORDS
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
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            return token[:-1]
        return token

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
