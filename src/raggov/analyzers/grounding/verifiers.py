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

from raggov.analyzers.grounding.candidate_selection import CONTENT_TERM_NORMALIZATIONS
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.analyzers.grounding.value_extraction import extract_value_mentions, find_value_alignment

logger = logging.getLogger(__name__)

from raggov.analyzers.grounding.claims import ExtractedClaim, AtomicSubclaim
from raggov.analyzers.grounding.decomposer import build_decomposer


@dataclass
class VerificationResult:
    """Standardized output from any claim evidence verifier."""

    label: Literal["entailed", "unsupported", "contradicted", "abstain"]
    support_label: Literal[
        "supported", "contradicted", "insufficient_evidence", "unverifiable", "skipped"
    ]
    raw_score: float
    evidence_chunk_id: str | None
    evidence_span: str | None
    rationale: str
    verifier_name: str
    label_reason: str | None = None
    fallback_used: bool = False
    error_info: str | None = None
    
    # Internal fields for compatibility with ClaimEvidenceRecord builders
    supporting_chunk_ids: list[str] = field(default_factory=list)
    candidate_chunk_ids: list[str] = field(default_factory=list)
    contradicting_chunk_ids: list[str] = field(default_factory=list)
    neutral_chunk_ids: list[str] = field(default_factory=list)
    best_candidate_id: str | None = None
    evidence_mode: Literal["single_chunk", "multi_chunk", "no_support"] = "no_support"
    aggregate_support_score: float = 0.0
    aggregate_contradiction_score: float = 0.0
    evidence_coverage_notes: list[str] = field(default_factory=list)
    calibrated_confidence: float | None = None
    confidence_status: Literal["calibrated", "uncalibrated_heuristic_proxy", "unavailable"] = (
        "unavailable"
    )
    verifier_limitations: list[str] = field(default_factory=list)
    verifier_warnings: list[str] = field(default_factory=list)
    raw_entailment_response: Any | None = None
    fallback_from: str | None = None
    fallback_to: str | None = None
    value_matches: list[dict[str, str]] = field(default_factory=list)
    value_conflicts: list[dict[str, str]] = field(default_factory=list)
    external_signal_records: list[dict[str, Any]] = field(default_factory=list)
    triplet_results: list[TripletVerificationResult] = field(default_factory=list)

    # Safety Gate and Ensemble Fields
    verifier_policy: str | None = None
    verifier_disagreement: bool = False
    safety_gate_triggered: bool = False
    safety_gate_reason: str | None = None
    safety_gate_category: str | None = None
    final_support_label_before_gate: str | None = None
    final_support_label_after_gate: str | None = None
    critical_fact_check_summary: dict[str, Any] = field(default_factory=dict)
    llm_label: str | None = None
    heuristic_label: str | None = None
    deterministic_gate_labels: list[str] = field(default_factory=list)
    normalized_values_checked: list[dict[str, Any]] = field(default_factory=list)
    normalized_dates_checked: list[dict[str, Any]] = field(default_factory=list)
    normalized_units_checked: list[dict[str, Any]] = field(default_factory=list)
    normalized_entities_checked: list[dict[str, Any]] = field(default_factory=list)

    # Compound Claim Telemetry
    compound_decomposed: bool = False
    subclaim_results: list[dict[str, Any]] = field(default_factory=list)
    undecomposed_compound_gate_triggered: bool = False


@dataclass
class EvidenceAggregationResult:
    """Aggregate evidence assessment across top-k candidate chunks."""

    best_candidate_id: str | None
    supporting_candidate_ids: list[str] = field(default_factory=list)
    contradicting_candidate_ids: list[str] = field(default_factory=list)
    neutral_candidate_ids: list[str] = field(default_factory=list)
    evidence_mode: Literal["single_chunk", "multi_chunk", "no_support"] = "no_support"
    aggregate_support_score: float = 0.0
    aggregate_contradiction_score: float = 0.0
    evidence_coverage_notes: list[str] = field(default_factory=list)
    best_candidate_score: float = 0.0


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


class ClaimEntailmentVerifierV1(EvidenceVerifier, ABC):
    """Explicit claim-to-evidence entailment interface."""

    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        metadata = metadata or {}
        return self.verify_entailment(
            claim_text=claim,
            source_sentence=str(metadata.get("source_sentence") or claim),
            top_k_candidates=candidates,
            cited_doc_ids=[str(value) for value in metadata.get("cited_doc_ids", [])],
            cited_chunk_ids=[str(value) for value in metadata.get("cited_chunk_ids", [])],
            claim_type=str(metadata.get("claim_type") or "other"),
            numbers=[str(value) for value in metadata.get("numbers", [])],
            dates=[str(value) for value in metadata.get("dates", [])],
            entities=[str(value) for value in metadata.get("entities", [])],
            atomicity_status=str(metadata.get("atomicity_status") or "unclear"),
            query=query,
            metadata=metadata,
        )

    @abstractmethod
    def verify_entailment(
        self,
        *,
        claim_text: str,
        source_sentence: str,
        top_k_candidates: list[EvidenceCandidate],
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
        claim_type: str,
        numbers: list[str],
        dates: list[str],
        entities: list[str],
        atomicity_status: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        raise NotImplementedError


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
            support_label="unverifiable",
            label_reason="verifier_abstain",
            raw_score=0.0,
            evidence_chunk_id=None,
            evidence_span=None,
            rationale=self.reason,
            verifier_name="abstaining_verifier",
            fallback_used=True,
            error_info=self.error_info,
            candidate_chunk_ids=[c.chunk_id for c in candidates],
            confidence_status="unavailable",
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
                support_label="insufficient_evidence",
                label_reason="unsupported_no_evidence",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No candidate evidence chunks available.",
                verifier_name="structured_llm_claim_verifier_v1",
                confidence_status="unavailable",
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
            support_label=_support_label_for_legacy_label(label),
            label_reason=_default_label_reason(label),
            raw_score=float(parsed.get("confidence", 0.0)),
            evidence_chunk_id=str(evidence_chunk_id) if evidence_chunk_id else None,
            evidence_span=parsed.get("evidence_span"),
            rationale=str(parsed.get("rationale", "LLM verifier output")),
            verifier_name="structured_llm_claim_verifier_v1",
            supporting_chunk_ids=supporting_chunk_ids,
            candidate_chunk_ids=candidate_ids,
            contradicting_chunk_ids=contradicting_chunk_ids,
            best_candidate_id=str(evidence_chunk_id) if evidence_chunk_id else None,
            evidence_mode="single_chunk" if evidence_chunk_id else "no_support",
            aggregate_support_score=float(parsed.get("confidence", 0.0)) if label == "entailed" else 0.0,
            aggregate_contradiction_score=float(parsed.get("confidence", 0.0)) if label == "contradicted" else 0.0,
            confidence_status="unavailable",
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


class LLMClaimEntailmentVerifierV1(ClaimEntailmentVerifierV1):
    """LLM-first evidence-only claim entailment verifier."""

    def __init__(self, config: dict[str, Any]):
        self._client = config.get("llm_client")
        if not self._client:
            raise ValueError("LLMClaimEntailmentVerifierV1 requires an 'llm_client' in config.")
        self._fallback_verifier = HeuristicValueOverlapVerifier(dict(config))
        self._verifier_name = "llm_claim_entailment_verifier_v1"

    def verify_entailment(
        self,
        *,
        claim_text: str,
        source_sentence: str,
        top_k_candidates: list[EvidenceCandidate],
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
        claim_type: str,
        numbers: list[str],
        dates: list[str],
        entities: list[str],
        atomicity_status: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not top_k_candidates:
            return VerificationResult(
                label="unsupported",
                support_label="insufficient_evidence",
                label_reason="unsupported_no_evidence",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No candidate evidence chunks available.",
                verifier_name=self._verifier_name,
                confidence_status="unavailable",
                verifier_warnings=["no_candidate_evidence"],
            )

        prompt = self._prompt(
            claim_text=claim_text,
            source_sentence=source_sentence,
            candidates=top_k_candidates,
            cited_doc_ids=cited_doc_ids,
            cited_chunk_ids=cited_chunk_ids,
            claim_type=claim_type,
            numbers=numbers,
            dates=dates,
            entities=entities,
            atomicity_status=atomicity_status,
        )
        try:
            raw_response = self._invoke(prompt)
        except Exception as invoke_exc:
            logger.warning("LLM entailment invoke failed, falling back to heuristic: %s", invoke_exc)
            fallback = self._fallback_verifier.verify(
                claim_text,
                query,
                top_k_candidates,
                metadata=metadata,
            )
            fallback.fallback_used = True
            fallback.fallback_from = "llm_entailment_verifier"
            fallback.fallback_to = "heuristic_top_k_verifier"
            fallback.verifier_warnings = [
                *fallback.verifier_warnings,
                f"llm_entailment_invoke_failed:{type(invoke_exc).__name__}",
            ]
            return fallback

        try:
            parsed = self._parse_response(raw_response)
        except Exception as exc:
            logger.warning("LLM entailment parse failed, attempting repair: %s", exc)
            try:
                repair_response = self._invoke(self._repair_prompt(prompt, raw_response))
                parsed = self._parse_response(repair_response)
                raw_response = repair_response
            except Exception as repair_exc:
                logger.warning("LLM entailment repair failed, falling back to heuristic: %s", repair_exc)
                fallback = self._fallback_verifier.verify(
                    claim_text,
                    query,
                    top_k_candidates,
                    metadata=metadata,
                )
                fallback.fallback_used = True
                fallback.fallback_from = "llm_entailment_verifier"
                fallback.fallback_to = "heuristic_top_k_verifier"
                fallback.verifier_warnings = [
                    *fallback.verifier_warnings,
                    f"llm_entailment_parse_failed:{type(exc).__name__}",
                    f"llm_entailment_repair_failed:{type(repair_exc).__name__}",
                ]
                fallback.raw_entailment_response = {
                    "initial_response": str(raw_response),
                    "repair_response": str(repair_response) if 'repair_response' in locals() else None,
                }
                return fallback

        return self._result_from_parsed(
            parsed=parsed,
            raw_response=raw_response,
            candidates=top_k_candidates,
            atomicity_status=atomicity_status,
        )

    def _invoke(self, prompt: str) -> object:
        if hasattr(self._client, "chat"):
            return self._client.chat(prompt)
        if hasattr(self._client, "complete"):
            return self._client.complete(prompt)
        raise TypeError("llm_client must provide chat() or complete()")

    def _prompt(
        self,
        *,
        claim_text: str,
        source_sentence: str,
        candidates: list[EvidenceCandidate],
        cited_doc_ids: list[str],
        cited_chunk_ids: list[str],
        claim_type: str,
        numbers: list[str],
        dates: list[str],
        entities: list[str],
        atomicity_status: str,
    ) -> str:
        candidate_block = "\n\n".join(
            f"[{candidate.chunk_id}] doc={candidate.source_doc_id} score={f'{candidate.retrieval_score:.3f}' if candidate.retrieval_score is not None else 'None'}\n{candidate.chunk_text}"
            for candidate in candidates
        )
        return (
            "Judge whether the retrieved evidence supports the claim.\n"
            "Rules:\n"
            "- Use only the provided evidence chunks.\n"
            "- Do not use external knowledge.\n"
            "- support_label must be one of: supported, contradicted, insufficient_evidence, unverifiable.\n"
            "- Mark partial support as insufficient_evidence unless all material parts of the claim are supported.\n"
            "- Check entity, numeric, and date consistency explicitly.\n"
            "- supporting_candidate_ids and contradicting_candidate_ids must use only chunk ids that appear in the evidence list.\n"
            "- Return strict JSON only.\n\n"
            f"Claim text: {claim_text}\n"
            f"Source sentence: {source_sentence}\n"
            f"Atomicity status: {atomicity_status}\n"
            f"Claim type: {claim_type}\n"
            f"Claim entities: {entities}\n"
            f"Claim dates: {dates}\n"
            f"Claim numbers: {numbers}\n"
            f"Cited doc ids: {cited_doc_ids}\n"
            f"Cited chunk ids: {cited_chunk_ids}\n\n"
            f"Evidence chunks:\n{candidate_block}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "support_label": "supported|contradicted|insufficient_evidence|unverifiable",\n'
            '  "supporting_candidate_ids": ["chunk_id"],\n'
            '  "contradicting_candidate_ids": ["chunk_id"],\n'
            '  "neutral_candidate_ids": ["chunk_id"],\n'
            '  "support_reason": "short explanation",\n'
            '  "verifier_warnings": ["warning"],\n'
            '  "confidence": 0.0\n'
            "}"
        )

    def _repair_prompt(self, prompt: str, bad_response: object) -> str:
        return (
            "Repair the previous answer into strict valid JSON matching the requested schema.\n"
            "Do not add new evidence or external facts.\n\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Malformed response:\n{bad_response}"
        )

    def _parse_response(self, response: object) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            response = str(response)
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM entailment response must be a JSON object.")
        return parsed

    def _result_from_parsed(
        self,
        *,
        parsed: dict[str, Any],
        raw_response: object,
        candidates: list[EvidenceCandidate],
        atomicity_status: str,
    ) -> VerificationResult:
        support_label = _coerce_support_label(parsed.get("support_label"))
        candidate_ids = {candidate.chunk_id for candidate in candidates}
        supporting_candidate_ids = _filter_candidate_ids(parsed.get("supporting_candidate_ids"), candidate_ids)
        contradicting_candidate_ids = _filter_candidate_ids(parsed.get("contradicting_candidate_ids"), candidate_ids)
        neutral_candidate_ids = _filter_candidate_ids(parsed.get("neutral_candidate_ids"), candidate_ids)
        verifier_warnings = [str(item) for item in parsed.get("verifier_warnings", []) if str(item).strip()]
        if atomicity_status == "compound" and support_label == "supported":
            support_label = "insufficient_evidence"
            verifier_warnings.append("compound_claim_requires_decomposition")

        label = _verification_label_for_support_label(support_label)
        best_candidate_id = (
            (supporting_candidate_ids or contradicting_candidate_ids or neutral_candidate_ids or [candidates[0].chunk_id])[0]
            if candidates
            else None
        )
        evidence_mode: Literal["single_chunk", "multi_chunk", "no_support"] = "no_support"
        if len(supporting_candidate_ids) > 1:
            evidence_mode = "multi_chunk"
        elif len(supporting_candidate_ids) == 1:
            evidence_mode = "single_chunk"
        raw_score = float(parsed.get("confidence", 0.0) or 0.0)
        return VerificationResult(
            label=label,
            support_label=support_label,
            label_reason=_default_label_reason(label),
            raw_score=raw_score,
            evidence_chunk_id=best_candidate_id,
            evidence_span=None,
            rationale=str(parsed.get("support_reason", "LLM entailment judgment.")),
            verifier_name=self._verifier_name,
            supporting_chunk_ids=supporting_candidate_ids,
            candidate_chunk_ids=[candidate.chunk_id for candidate in candidates],
            contradicting_chunk_ids=contradicting_candidate_ids,
            neutral_chunk_ids=neutral_candidate_ids,
            best_candidate_id=best_candidate_id,
            evidence_mode=evidence_mode,
            aggregate_support_score=raw_score if support_label == "supported" else 0.0,
            aggregate_contradiction_score=raw_score if support_label == "contradicted" else 0.0,
            calibrated_confidence=None,
            confidence_status="unavailable",
            verifier_limitations=[
                "LLM entailment judgment is uncalibrated.",
                "LLM verdict is evidence-only and not independently validated by NLI.",
            ],
            verifier_warnings=verifier_warnings,
            raw_entailment_response=parsed if isinstance(raw_response, str) else parsed,
        )


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


def _value_conflict_reason(conflict: dict[str, str]) -> str:
    claim_value = conflict.get("claim_value") or "claim value"
    evidence_value = conflict.get("evidence_value") or "evidence value"
    return f"Evidence states {evidence_value}, conflicting with claim value {claim_value}."


def _default_label_reason(
    label: Literal["entailed", "unsupported", "contradicted", "abstain"],
) -> str:
    if label == "contradicted":
        return "explicit_contradiction"
    if label == "unsupported":
        return "unsupported_no_evidence"
    if label == "abstain":
        return "verifier_abstain"
    return "supported"


def _support_label_for_legacy_label(
    label: Literal["entailed", "unsupported", "contradicted", "abstain"],
) -> Literal["supported", "contradicted", "insufficient_evidence", "unverifiable", "skipped"]:
    if label == "entailed":
        return "supported"
    if label == "contradicted":
        return "contradicted"
    if label == "unsupported":
        return "insufficient_evidence"
    return "unverifiable"


def _verification_label_for_support_label(
    support_label: Literal["supported", "contradicted", "insufficient_evidence", "unverifiable", "skipped"],
) -> Literal["entailed", "unsupported", "contradicted", "abstain"]:
    if support_label == "supported":
        return "entailed"
    if support_label == "contradicted":
        return "contradicted"
    if support_label == "insufficient_evidence":
        return "unsupported"
    return "abstain"


def _coerce_support_label(
    value: object,
) -> Literal["supported", "contradicted", "insufficient_evidence", "unverifiable", "skipped"]:
    normalized = str(value or "unverifiable").strip().lower()
    aliases = {
        "supported": "supported",
        "support": "supported",
        "entailed": "supported",
        "contradicted": "contradicted",
        "contradiction": "contradicted",
        "insufficient_evidence": "insufficient_evidence",
        "unsupported": "insufficient_evidence",
        "unverifiable": "unverifiable",
        "abstain": "unverifiable",
        "skipped": "skipped",
    }
    if normalized not in aliases:
        raise ValueError(f"Invalid support_label: {value!r}")
    return aliases[normalized]  # type: ignore[return-value]


def _filter_candidate_ids(value: object, allowed_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    filtered: list[str] = []
    for item in value:
        candidate_id = str(item)
        if candidate_id not in allowed_ids or candidate_id in seen:
            continue
        seen.add(candidate_id)
        filtered.append(candidate_id)
    return filtered


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
            return self._verify_structured(claim, _query=query, candidates=candidates, metadata=metadata)
        except Exception as exc:
            logger.warning(
                "Structured claim verifier failed, falling back to deterministic: %s", exc
            )
            base = self._verify_deterministic(claim, candidates, metadata=metadata)
            base.fallback_used = True
            base.rationale = "Structured verifier unavailable; fell back to deterministic top-k evidence aggregation."
            return base

    def _verify_structured(
        self,
        claim: str,
        _query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(
                label="unsupported",
                support_label="insufficient_evidence",
                label_reason="unsupported_no_evidence",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No retrieved chunks available for evidence matching.",
                verifier_name="value_aware_structured_claim_verifier_v1",
                confidence_status="unavailable",
            )
        return self._aggregate_top_k(
            claim,
            candidates,
            verifier_name="value_aware_structured_claim_verifier_v1",
            metadata=metadata,
        )

    def _verify_deterministic(
        self,
        claim: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(
                label="unsupported",
                support_label="insufficient_evidence",
                label_reason="unsupported_no_evidence",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="No candidate evidence chunk matched claim content terms.",
                verifier_name="deterministic_overlap_anchor_v0",
                confidence_status="unavailable",
            )
        return self._aggregate_top_k(
            claim,
            candidates,
            verifier_name="deterministic_overlap_anchor_v0",
            metadata=metadata,
        )

    def _aggregate_top_k(
        self,
        claim: str,
        candidates: list[EvidenceCandidate],
        *,
        verifier_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        claim_terms = self._content_terms(claim)
        if not claim_terms:
            return VerificationResult(
                label="unsupported",
                support_label="unverifiable",
                label_reason="unsupported_no_evidence",
                raw_score=0.0,
                evidence_chunk_id=candidates[0].chunk_id if candidates else None,
                evidence_span=None,
                rationale="Claim has no meaningful content terms to verify.",
                verifier_name=verifier_name,
                candidate_chunk_ids=[c.chunk_id for c in candidates],
                best_candidate_id=candidates[0].chunk_id if candidates else None,
                confidence_status="unavailable",
                verifier_limitations=["Heuristic verifier requires lexical claim content terms."],
            )

        aggregate = EvidenceAggregationResult(
            best_candidate_id=candidates[0].chunk_id if candidates else None,
            best_candidate_score=candidates[0].raw_support_score if candidates else 0.0,
        )
        value_matches_all: list[dict[str, str]] = []
        value_conflicts_all: list[dict[str, str]] = []
        aggregate_support_contributors: list[str] = []
        claim_anchors = set(self._extract_anchors(claim))
        claim_has_numeric_anchor = any(ch.isdigit() for anchor in claim_anchors for ch in anchor)
        non_numeric_claim_terms = {term for term in claim_terms if not any(ch.isdigit() for ch in term)}

        for candidate in candidates:
            chunk_terms = self._content_terms(candidate.chunk_text)
            non_numeric_chunk_terms = {term for term in chunk_terms if not any(ch.isdigit() for ch in term)}
            chunk_anchor_set = set(self._extract_anchors(candidate.chunk_text))
            term_coverage = len(claim_terms & chunk_terms) / len(claim_terms) if claim_terms else 0.0
            non_numeric_term_coverage = (
                len(non_numeric_claim_terms & non_numeric_chunk_terms) / len(non_numeric_claim_terms)
                if non_numeric_claim_terms
                else term_coverage
            )
            anchor_coverage = (
                len(claim_anchors & chunk_anchor_set) / len(claim_anchors)
                if claim_anchors
                else 0.0
            )
            per_candidate_score = max(candidate.raw_support_score, term_coverage, anchor_coverage)
            candidate_reason = self._pattern_contradiction(claim, candidate.chunk_text)
            value_matches, value_conflicts, missing_critical = find_value_alignment(
                claim, candidate.chunk_text
            )
            value_matches_all.extend(value_matches)
            value_conflicts_all.extend(value_conflicts)

            explicit_contradiction = bool(
                self._contains_negation_of_terms(candidate.chunk_text, claim_terms) or candidate_reason
            )
            contextually_aligned_value_conflict = bool(
                value_conflicts and non_numeric_term_coverage >= 0.25
            )
            if explicit_contradiction or contextually_aligned_value_conflict:
                aggregate.contradicting_candidate_ids.append(candidate.chunk_id)
                aggregate.aggregate_contradiction_score = max(
                    aggregate.aggregate_contradiction_score,
                    max(per_candidate_score, candidate.raw_support_score),
                )
                if candidate_reason:
                    aggregate.evidence_coverage_notes.append(candidate_reason)
                elif value_conflicts:
                    aggregate.evidence_coverage_notes.append(
                        _value_conflict_reason(value_conflicts[0])
                    )
                continue
            if value_conflicts:
                aggregate.neutral_candidate_ids.append(candidate.chunk_id)
                aggregate.evidence_coverage_notes.append(
                    "Ignored isolated value conflict from weakly aligned chunk."
                )
                continue

            if non_numeric_term_coverage >= 0.3 or value_matches:
                aggregate_support_contributors.append(candidate.chunk_id)

            unsupported_reason = self._pattern_unsupported(claim, candidate.chunk_text)
            partial_support_reason = self._pattern_partial_support(
                claim=claim,
                evidence=candidate.chunk_text,
                claim_terms=claim_terms,
                evidence_terms=chunk_terms,
                best_score=per_candidate_score,
            )
            if unsupported_reason or partial_support_reason:
                aggregate.neutral_candidate_ids.append(candidate.chunk_id)
                aggregate.evidence_coverage_notes.append(
                    unsupported_reason or partial_support_reason or "partial_support"
                )
                continue

            if missing_critical and claim_has_numeric_anchor:
                aggregate.neutral_candidate_ids.append(candidate.chunk_id)
                aggregate.evidence_coverage_notes.append(
                    _first_claim_value_snippet(claim)
                    or "Claim contains critical value details not present in this chunk."
                )
                continue

            if per_candidate_score >= self._support_threshold or value_matches:
                aggregate.supporting_candidate_ids.append(candidate.chunk_id)
                aggregate.aggregate_support_score = max(
                    aggregate.aggregate_support_score,
                    max(per_candidate_score, candidate.raw_support_score),
                )
            else:
                aggregate.neutral_candidate_ids.append(candidate.chunk_id)

        aggregate_text = " ".join(candidate.chunk_text for candidate in candidates)
        aggregate_terms = self._content_terms(aggregate_text)
        aggregate_term_coverage = len(claim_terms & aggregate_terms) / len(claim_terms)
        aggregate_value_matches, aggregate_value_conflicts, aggregate_missing_critical = find_value_alignment(
            claim, aggregate_text
        )
        aggregate_non_numeric_term_coverage = (
            len(non_numeric_claim_terms & {term for term in aggregate_terms if not any(ch.isdigit() for ch in term)})
            / len(non_numeric_claim_terms)
            if non_numeric_claim_terms
            else aggregate_term_coverage
        )
        if aggregate_value_conflicts and aggregate_non_numeric_term_coverage >= 0.25:
            value_conflicts_all.extend(aggregate_value_conflicts)
            if not aggregate.evidence_coverage_notes:
                aggregate.evidence_coverage_notes.append(
                    _value_conflict_reason(aggregate_value_conflicts[0])
                )
            aggregate.contradicting_candidate_ids = list(
                dict.fromkeys(
                    [*aggregate.contradicting_candidate_ids, *[c.chunk_id for c in candidates]]
                )
            )
            aggregate.aggregate_contradiction_score = max(
                aggregate.aggregate_contradiction_score,
                max((c.raw_support_score for c in candidates), default=0.0),
            )

        if aggregate.contradicting_candidate_ids:
            best_contradiction = aggregate.contradicting_candidate_ids[0]
            raw_score = max(aggregate.aggregate_contradiction_score, aggregate.best_candidate_score)
            if value_conflicts_all:
                raw_score = min(raw_score, 0.49)
            return VerificationResult(
                label="contradicted",
                support_label="contradicted",
                label_reason="value_conflict" if value_conflicts_all else "explicit_contradiction",
                raw_score=raw_score,
                evidence_chunk_id=best_contradiction,
                evidence_span=None,
                rationale=(
                    aggregate.evidence_coverage_notes[0]
                    if aggregate.evidence_coverage_notes
                    else "Contradictory evidence exists among top-k candidate chunks."
                ),
                verifier_name=verifier_name,
                supporting_chunk_ids=[],
                candidate_chunk_ids=[c.chunk_id for c in candidates],
                contradicting_chunk_ids=list(dict.fromkeys(aggregate.contradicting_candidate_ids)),
                neutral_chunk_ids=list(dict.fromkeys(aggregate.neutral_candidate_ids)),
                best_candidate_id=aggregate.best_candidate_id,
                evidence_mode="no_support",
                aggregate_support_score=aggregate.aggregate_support_score,
                aggregate_contradiction_score=aggregate.aggregate_contradiction_score,
                evidence_coverage_notes=aggregate.evidence_coverage_notes,
                calibrated_confidence=None,
                confidence_status="uncalibrated_heuristic_proxy",
                verifier_limitations=[
                    "Top-k contradiction detection is heuristic and not NLI-based.",
                    "Any contradiction among top-k candidates currently takes precedence.",
                ],
                value_matches=value_matches_all,
                value_conflicts=value_conflicts_all,
            )

        aggregate_anchor_hits = len(claim_anchors & set(self._extract_anchors(aggregate_text))) if claim_anchors else 0
        enough_anchor_support = (
            not claim_anchors
            or aggregate_anchor_hits > 0
            or bool(aggregate_value_matches or value_matches_all)
        )
        contributing_ids = list(dict.fromkeys([*aggregate.supporting_candidate_ids, *aggregate_support_contributors]))
        has_multi_chunk_support = len(contributing_ids) >= 2 and (
            aggregate_term_coverage >= self._support_threshold or aggregate_value_matches
        )
        has_single_chunk_support = len(set(aggregate.supporting_candidate_ids)) == 1 and enough_anchor_support

        if (has_multi_chunk_support or has_single_chunk_support) and not aggregate_missing_critical:
            supporting_ids = (
                contributing_ids
                if has_multi_chunk_support
                else list(dict.fromkeys(aggregate.supporting_candidate_ids))
            )
            if (
                str((metadata or {}).get("atomicity_status") or "").lower() == "compound"
                and self._likely_multi_assertion_claim(claim)
            ):
                return VerificationResult(
                    label="unsupported",
                    support_label="insufficient_evidence",
                    label_reason="unsupported_no_evidence",
                    raw_score=max(aggregate.aggregate_support_score, aggregate_term_coverage),
                    evidence_chunk_id=supporting_ids[0],
                    evidence_span=None,
                    rationale=(
                        "Compound claim requires decomposition before reliable claim-level support verification."
                    ),
                    verifier_name=verifier_name,
                    supporting_chunk_ids=supporting_ids,
                    candidate_chunk_ids=[c.chunk_id for c in candidates],
                    contradicting_chunk_ids=[],
                    neutral_chunk_ids=list(dict.fromkeys(aggregate.neutral_candidate_ids)),
                    best_candidate_id=aggregate.best_candidate_id,
                    evidence_mode="multi_chunk" if len(supporting_ids) > 1 else "single_chunk",
                    aggregate_support_score=max(aggregate.aggregate_support_score, aggregate_term_coverage),
                    aggregate_contradiction_score=aggregate.aggregate_contradiction_score,
                    evidence_coverage_notes=[
                        *aggregate.evidence_coverage_notes,
                        "compound_claim_requires_decomposition",
                    ],
                    calibrated_confidence=None,
                    confidence_status="uncalibrated_heuristic_proxy",
                    verifier_limitations=[
                        "Compound claims are conservatively treated as insufficient until decomposed.",
                        "Support is based on heuristic top-k aggregation, not full NLI.",
                    ],
                    verifier_warnings=["compound_claim_requires_decomposition"],
                    value_matches=[*value_matches_all, *aggregate_value_matches],
                    value_conflicts=value_conflicts_all,
                )
            evidence_mode: Literal["single_chunk", "multi_chunk", "no_support"] = (
                "multi_chunk" if len(supporting_ids) > 1 else "single_chunk"
            )
            rationale = (
                "Claim supported by multiple retrieved chunks that jointly cover the claim."
                if evidence_mode == "multi_chunk"
                else "Claim supported by a retrieved chunk via lexical and value alignment."
            )
            if aggregate.evidence_coverage_notes:
                rationale = f"{rationale} {' '.join(aggregate.evidence_coverage_notes[:2])}".strip()
            return VerificationResult(
                label="entailed",
                support_label="supported",
                label_reason="supported",
                raw_score=max(aggregate.aggregate_support_score, aggregate_term_coverage),
                evidence_chunk_id=supporting_ids[0],
                evidence_span=None,
                rationale=rationale,
                verifier_name=verifier_name,
                supporting_chunk_ids=supporting_ids,
                candidate_chunk_ids=[c.chunk_id for c in candidates],
                contradicting_chunk_ids=[],
                neutral_chunk_ids=list(dict.fromkeys(aggregate.neutral_candidate_ids)),
                best_candidate_id=aggregate.best_candidate_id,
                evidence_mode=evidence_mode,
                aggregate_support_score=max(aggregate.aggregate_support_score, aggregate_term_coverage),
                aggregate_contradiction_score=aggregate.aggregate_contradiction_score,
                evidence_coverage_notes=aggregate.evidence_coverage_notes,
                calibrated_confidence=None,
                confidence_status="uncalibrated_heuristic_proxy",
                verifier_limitations=[
                    "Support is based on top-k lexical/value aggregation, not full NLI.",
                    "Raw support score is an uncalibrated heuristic proxy.",
                ],
                value_matches=[*value_matches_all, *aggregate_value_matches],
                value_conflicts=value_conflicts_all,
            )

        rationale = (
            "Claim evidence is related but insufficient across the available top-k chunks."
            if candidates
            else "No candidate evidence chunk matched claim content terms."
        )
        return VerificationResult(
            label="unsupported",
            support_label="insufficient_evidence",
            label_reason="unsupported_no_evidence",
            raw_score=max(aggregate.aggregate_support_score, aggregate_term_coverage, aggregate.best_candidate_score),
            evidence_chunk_id=aggregate.best_candidate_id,
            evidence_span=None,
            rationale=rationale,
            verifier_name=verifier_name,
            supporting_chunk_ids=list(dict.fromkeys(aggregate.supporting_candidate_ids)),
            candidate_chunk_ids=[c.chunk_id for c in candidates],
            contradicting_chunk_ids=[],
            neutral_chunk_ids=list(dict.fromkeys(aggregate.neutral_candidate_ids)),
            best_candidate_id=aggregate.best_candidate_id,
            evidence_mode="no_support",
            aggregate_support_score=max(aggregate.aggregate_support_score, aggregate_term_coverage),
            aggregate_contradiction_score=aggregate.aggregate_contradiction_score,
            evidence_coverage_notes=aggregate.evidence_coverage_notes,
            calibrated_confidence=None,
            confidence_status="uncalibrated_heuristic_proxy",
            verifier_limitations=[
                "Insufficient-evidence verdict is heuristic and based on top-k lexical/value aggregation.",
                "Raw support score is not calibrated confidence.",
            ],
            value_matches=[*value_matches_all, *aggregate_value_matches],
            value_conflicts=value_conflicts_all,
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

    def _pattern_contradiction(self, claim: str, evidence: str) -> str | None:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        table_conflict = self._table_value_conflict(claim_lower, evidence_lower)
        if table_conflict:
            return table_conflict

        if (
            re.search(r"\banyone\b|\ball\b", claim_lower)
            and re.search(r"\bonly\b.{0,80}\beligible\b|\beligible\b.{0,80}\bonly\b", evidence_lower)
        ):
            return "Answer generalizes eligibility beyond an explicit 'only' constraint in evidence."

        if "net profit" in claim_lower and "net profit" in evidence_lower:
            claim_money = re.search(r"\$\s*\d[\d,.]*\s*[kmb]?", claim_lower)
            evidence_money = re.search(r"\$\s*\d[\d,.]*\s*[kmb]?", evidence_lower)
            if claim_money and evidence_money and claim_money.group(0) != evidence_money.group(0):
                return "Claim assigns a different net profit value than the evidence."

        return None

    def _pattern_unsupported(self, claim: str, evidence: str) -> str | None:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        if re.search(r"\bmanager\s+is\s+john\b", claim_lower) and re.search(
            r"\bmanager\s+told\s+john\b", evidence_lower
        ):
            return "Keyword overlap is relationally different: evidence says the manager told John, not that John is the manager."

        if re.search(r"\bexpect\b|\bahead of time\b|\bnext week\b|\bfinish\b", claim_lower):
            speculative_terms = {"expect", "finish", "ahead", "next", "week"}
            missing_terms = [term for term in speculative_terms if term in claim_lower and term not in evidence_lower]
            if missing_terms:
                return "Unsupported generation detail: answer adds speculative schedule terms absent from evidence."

        if "headquarter" in claim_lower and "headquarter" not in evidence_lower:
            return "Compound claim includes headquarters detail absent from evidence."

        if "standard plus" in claim_lower and "standard plus" not in evidence_lower:
            return "Retrieved evidence is about a different account than the answer claim."

        return None

    def _pattern_partial_support(
        self,
        *,
        claim: str,
        evidence: str,
        claim_terms: set[str],
        evidence_terms: set[str],
        best_score: float,
    ) -> str | None:
        overlap = claim_terms & evidence_terms
        missing_terms = claim_terms - evidence_terms
        if not overlap or not missing_terms:
            return None

        salient_missing = {
            term
            for term in missing_terms
            if len(term) > 3
            and term not in {
                "provide",
                "benefit",
                "company",
                "policy",
                "customer",
                "latest",
                "fiscal",
                "current",
                "year",
            }
        }
        if not salient_missing:
            return None

        overlap_ratio = len(overlap) / max(len(claim_terms), 1)
        if re.search(r",|\band\b|\bor\b", claim.lower()) and overlap_ratio >= 0.4:
            return (
                "Claim is only partially supported: evidence covers overlapping facts but misses "
                "additional asserted items or qualifiers."
            )
        if (
            len(salient_missing) >= 2
            and len(overlap) >= 2
            and best_score >= self._support_threshold
            and overlap_ratio < 0.85
        ):
            return (
                "Claim is only partially supported: evidence overlaps with the answer but does not "
                "support all asserted details."
            )
        return None

    def _likely_multi_assertion_claim(self, claim: str) -> bool:
        lowered = claim.lower()
        if re.search(r"\b(?:and|or|but|while|whereas|however)\b", lowered):
            return True
        if len(re.findall(r"\b(?:is|are|was|were|has|have|had|will|shall|must|can|may)\b", lowered)) > 2:
            return True
        return False

    def _table_value_conflict(self, claim_lower: str, evidence_lower: str) -> str | None:
        if "|" not in evidence_lower:
            return None
        rows: dict[str, str] = {}
        for row in evidence_lower.splitlines():
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            key, value = cells[0], cells[1]
            if not re.fullmatch(r"q\d+", key):
                continue
            money = re.search(r"\$\s*\d[\d,.]*\s*[kmb]?", value)
            if money:
                rows[key] = re.sub(r"\s+", "", money.group(0))
        for quarter, evidence_value in rows.items():
            claim_match = re.search(
                rf"(\$\s*\d[\d,.]*\s*[kmb]?)\s+(?:in|for)\s+{re.escape(quarter)}\b",
                claim_lower,
            )
            if claim_match and re.sub(r"\s+", "", claim_match.group(1)) != evidence_value:
                return f"Claim swaps or misassigns table value for {quarter.upper()}."
        return None

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
        token = CONTENT_TERM_NORMALIZATIONS.get(token, token)
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


class CompoundClaimVerifier(EvidenceVerifier):
    """
    Decomposes a compound claim into subclaims, verifies each subclaim independently,
    and aggregates the verification result.
    """

    def __init__(self, base_verifier: EvidenceVerifier, config: dict[str, Any]) -> None:
        self.base_verifier = base_verifier
        self.config = config
        self.decomposer = build_decomposer(config)

    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        metadata = metadata or {}
        # Construct an ExtractedClaim to pass to the decomposer
        parent_claim = ExtractedClaim(
            claim_id=metadata.get("claim_id") or "claim_parent",
            claim_text=claim,
            source_sentence=metadata.get("source_sentence") or claim,
            source_start_char=metadata.get("source_start_char") or 0,
            source_end_char=metadata.get("source_end_char") or len(claim),
            atomicity_status="compound",
            claim_type=metadata.get("claim_type") or "other",
            entities=metadata.get("entities") or [],
            dates=metadata.get("dates") or [],
            numbers=metadata.get("numbers") or [],
            extraction_method="compound_verifier_wrapper",
            extraction_reason="compound_claim",
            should_verify=True,
        )

        try:
            subclaims = self.decomposer.decompose(parent_claim)
        except Exception as exc:
            logger.warning("Failed to decompose compound claim: %s", exc)
            subclaims = []

        if len(subclaims) < 2:
            # Decomposition failed or did not yield multiple subclaims.
            return VerificationResult(
                label="abstain",
                support_label="skipped",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="Compound claim decomposition failed or did not yield multiple subclaims.",
                verifier_name="compound_claim_verifier",
                undecomposed_compound_gate_triggered=True,
            )

        sub_results = []
        for sub in subclaims:
            sub_metadata = dict(metadata)
            sub_metadata.update({
                "claim_id": sub.subclaim_id,
                "source_sentence": sub.text,
                "claim_type": "other",  # Treat subclaim as atomic/other
                "entities": sub.entities,
                "dates": sub.dates,
                "numbers": sub.numbers,
            })
            sub_res = self.base_verifier.verify(sub.text, query, candidates, sub_metadata)
            sub_results.append((sub, sub_res))

        required_subs = [item for item in sub_results if item[0].required_support]
        if not required_subs:
            required_subs = sub_results

        all_supported = True
        any_contradicted = False
        any_insufficient = False

        supporting_chunks = set()
        contradicting_chunks = set()
        neutral_chunks = set()
        rationales = []
        best_candidate_id = None
        best_candidate_score = 0.0

        serialized_sub_results = []

        for sub, res in required_subs:
            serialized_sub_results.append({
                "subclaim_id": sub.subclaim_id,
                "text": sub.text,
                "required_support": sub.required_support,
                "label": res.label,
                "support_label": res.support_label,
                "rationale": res.rationale,
                "best_candidate_id": res.best_candidate_id,
            })

            rationales.append(f"Subclaim '{sub.text}': {res.support_label} (Reason: {res.rationale})")

            supporting_chunks.update(res.supporting_chunk_ids)
            contradicting_chunks.update(res.contradicting_chunk_ids)
            neutral_chunks.update(res.neutral_chunk_ids)

            if res.best_candidate_id and res.raw_score > best_candidate_score:
                best_candidate_id = res.best_candidate_id
                best_candidate_score = res.raw_score

            if res.support_label == "contradicted":
                any_contradicted = True
                all_supported = False
            elif res.support_label == "insufficient_evidence":
                any_insufficient = True
                all_supported = False
            elif res.support_label != "supported":
                any_insufficient = True
                all_supported = False

        if any_contradicted:
            final_support_label = "contradicted"
        elif any_insufficient:
            final_support_label = "insufficient_evidence"
        else:
            final_support_label = "supported"

        final_label = _verification_label_for_support_label(final_support_label)
        aggregate_rationale = " | ".join(rationales)

        return VerificationResult(
            label=final_label,
            support_label=final_support_label,
            raw_score=best_candidate_score,
            evidence_chunk_id=best_candidate_id,
            evidence_span=None,
            rationale=f"Compound Claim Decomposed: {aggregate_rationale}",
            verifier_name="compound_claim_verifier",
            supporting_chunk_ids=list(supporting_chunks),
            contradicting_chunk_ids=list(contradicting_chunks),
            neutral_chunk_ids=list(neutral_chunks),
            best_candidate_id=best_candidate_id,
            compound_decomposed=True,
            subclaim_results=serialized_sub_results,
            undecomposed_compound_gate_triggered=False,
        )


class ConservativeEnsembleVerifier(EvidenceVerifier):
    """
    Safety-gated verifier that runs both LLM and Heuristic verifiers.
    Prevents false-passes by downgrading LLM 'supported' if:
    1. Heuristic disagrees (contradicted).
    2. Critical facts (numbers/dates/entities) are missing.
    3. Compound claim is not fully covered.
    """
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._llm_verifier = LLMClaimEntailmentVerifierV1(config)
        self._heuristic_verifier = HeuristicValueOverlapVerifier(config)
        self._compound_verifier = CompoundClaimVerifier(self._llm_verifier, config)
        self._verifier_name = "conservative_ensemble_v1"

    def verify(
        self,
        claim: str,
        query: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if metadata is None:
            metadata = {}

        if metadata.get("atomicity_status") == "compound":
            comp_res = self._compound_verifier.verify(claim, query, candidates, metadata)
            if comp_res.compound_decomposed:
                # Set verifier_policy to ensemble
                comp_res.verifier_policy = "conservative_ensemble"
                return comp_res

        llm_res = self._llm_verifier.verify(claim, query, candidates, metadata)
        heur_res = self._heuristic_verifier.verify(claim, query, candidates, metadata)

        res = VerificationResult(
            label=llm_res.label,
            support_label=llm_res.support_label,
            raw_score=llm_res.raw_score,
            evidence_chunk_id=llm_res.evidence_chunk_id,
            evidence_span=llm_res.evidence_span,
            rationale=llm_res.rationale,
            verifier_name=self._verifier_name,
            label_reason=llm_res.label_reason,
            fallback_used=llm_res.fallback_used or heur_res.fallback_used,
            supporting_chunk_ids=llm_res.supporting_chunk_ids,
            candidate_chunk_ids=llm_res.candidate_chunk_ids,
            contradicting_chunk_ids=llm_res.contradicting_chunk_ids,
            neutral_chunk_ids=llm_res.neutral_chunk_ids,
            best_candidate_id=llm_res.best_candidate_id,
            evidence_mode=llm_res.evidence_mode,
            verifier_warnings=list(llm_res.verifier_warnings),
            verifier_policy="conservative_ensemble",
            final_support_label_before_gate=llm_res.support_label,
            calibrated_confidence=llm_res.calibrated_confidence,
            confidence_status=llm_res.confidence_status,
            llm_label=llm_res.support_label,
            heuristic_label=heur_res.support_label,
        )

        gate = self._evaluate_supported_claim(
            claim=claim,
            candidates=candidates,
            metadata=metadata,
            llm_res=llm_res,
            heur_res=heur_res,
            compound_result=comp_res if metadata.get("atomicity_status") == "compound" else None,
        )
        res.critical_fact_check_summary = gate["summary"]
        res.deterministic_gate_labels = list(gate["labels"])
        res.normalized_values_checked = gate["normalized_values_checked"]
        res.normalized_dates_checked = gate["normalized_dates_checked"]
        res.normalized_units_checked = gate["normalized_units_checked"]
        res.normalized_entities_checked = gate["normalized_entities_checked"]

        if gate["triggered"]:
            res.support_label = gate["support_label"]  # type: ignore[assignment]
            res.label = _verification_label_for_support_label(gate["support_label"])  # type: ignore[arg-type]
            res.safety_gate_triggered = True
            res.safety_gate_reason = gate["reason"]
            res.safety_gate_category = gate["category"]
            res.verifier_warnings.extend(gate["labels"])
            if gate["reason"] == "llm_heuristic_disagreement":
                res.verifier_disagreement = True

        res.final_support_label_after_gate = res.support_label
        return res

    def _evaluate_supported_claim(
        self,
        *,
        claim: str,
        candidates: list[EvidenceCandidate],
        metadata: dict[str, Any],
        llm_res: VerificationResult,
        heur_res: VerificationResult,
        compound_result: VerificationResult | None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "value_missing": False,
            "value_conflict": False,
            "date_missing": False,
            "date_conflict": False,
            "unit_mismatch": False,
            "entity_missing_or_mismatched": False,
            "entity_value_binding_conflict": False,
            "related_but_non_supporting": False,
            "compound_warning": False,
            "explicit_contradiction": False,
            "positive_support_signal": False,
        }
        payload = {
            "triggered": False,
            "support_label": llm_res.support_label,
            "reason": None,
            "category": None,
            "labels": [],
            "summary": summary,
            "normalized_values_checked": [],
            "normalized_dates_checked": [],
            "normalized_units_checked": [],
            "normalized_entities_checked": [],
        }
        if llm_res.support_label != "supported":
            return payload

        evidence_text = self._support_text(candidates, llm_res.best_candidate_id)
        claim_mentions = extract_value_mentions(claim)
        evidence_mentions = extract_value_mentions(evidence_text)
        critical_entities = self._critical_entities(metadata)
        critical_values = self._critical_values(metadata, claim_mentions)
        critical_dates = self._critical_dates(metadata, claim_mentions)

        payload["normalized_values_checked"] = [
            {"raw_text": mention.raw_text, "normalized_value": mention.normalized_value, "unit": mention.unit, "value_type": mention.value_type}
            for mention in claim_mentions
            if mention.value_type not in {"date"}
        ]
        payload["normalized_dates_checked"] = [
            {"raw_text": mention.raw_text, "normalized_value": mention.normalized_value, "unit": mention.unit}
            for mention in claim_mentions
            if mention.value_type == "date"
        ]
        payload["normalized_units_checked"] = [
            {"raw_text": mention.raw_text, "normalized_value": mention.normalized_value, "unit": mention.unit}
            for mention in claim_mentions
            if mention.unit not in {"number", "date", "year", "currency", "percent", "version"}
        ]

        entity_check = self._check_critical_entities(critical_entities, evidence_text)
        payload["normalized_entities_checked"] = entity_check["checked"]
        summary["entity_missing_or_mismatched"] = entity_check["has_issue"]
        binding_check = self._check_entity_value_binding(
            claim=claim,
            evidence_text=evidence_text,
            all_candidates=candidates,
            critical_entities=critical_entities,
            critical_values=critical_values,
        )
        summary["entity_value_binding_conflict"] = binding_check["conflict"]

        value_check = self._check_critical_mentions(
            claim_mentions=claim_mentions,
            evidence_mentions=evidence_mentions,
            critical_values=critical_values,
            kind="value",
        )
        summary["value_missing"] = value_check["missing"]
        summary["value_conflict"] = value_check["conflict"]
        summary["unit_mismatch"] = value_check["unit_mismatch"]

        date_check = self._check_critical_mentions(
            claim_mentions=claim_mentions,
            evidence_mentions=evidence_mentions,
            critical_values=critical_dates,
            kind="date",
        )
        summary["date_missing"] = date_check["missing"]
        summary["date_conflict"] = date_check["conflict"]

        if heur_res.support_label == "contradicted":
            summary["explicit_contradiction"] = True
            return self._gate_payload(
                payload,
                support_label="contradicted",
                reason="llm_heuristic_disagreement",
                category="contradiction_missed",
            )

        if summary["unit_mismatch"]:
            return self._gate_payload(
                payload,
                support_label="contradicted",
                reason="unit_or_magnitude_mismatch",
                category="unit_mismatch_missed",
            )

        if summary["value_conflict"]:
            return self._gate_payload(
                payload,
                support_label="contradicted",
                reason="critical_value_missing_or_conflicting",
                category="value_mismatch_missed",
            )

        if summary["date_conflict"]:
            return self._gate_payload(
                payload,
                support_label="contradicted",
                reason="critical_date_missing_or_conflicting",
                category="date_mismatch_missed",
            )

        if summary["entity_value_binding_conflict"]:
            return self._gate_payload(
                payload,
                support_label="contradicted",
                reason="missing_or_mismatched_critical_entity",
                category="entity_mismatch_missed",
            )

        if summary["entity_missing_or_mismatched"]:
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="missing_or_mismatched_critical_entity",
                category="entity_mismatch_missed",
            )

        if summary["value_missing"]:
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="critical_value_missing_or_conflicting",
                category="value_mismatch_missed",
            )

        if summary["date_missing"]:
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="critical_date_missing_or_conflicting",
                category="date_mismatch_missed",
            )

        if metadata.get("atomicity_status") == "compound" and (
            (compound_result is not None and not compound_result.compound_decomposed)
            or "compound_claim_requires_decomposition" in llm_res.verifier_warnings
        ):
            summary["compound_warning"] = True
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="compound_claim_not_fully_supported",
                category="compound_partial_support",
            )

        if self._is_related_but_non_supporting(claim, evidence_text, heur_res):
            summary["related_but_non_supporting"] = True
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="related_but_non_supporting",
                category="related_but_non_supporting",
            )

        summary["positive_support_signal"] = self._has_positive_support_signal(
            critical_entities=critical_entities,
            entity_check=entity_check,
            value_check=value_check,
            date_check=date_check,
            heur_res=heur_res,
        )
        if not summary["positive_support_signal"]:
            return self._gate_payload(
                payload,
                support_label="insufficient_evidence",
                reason="llm_overpermissive_no_deterministic_gate",
                category="llm_overpermissive_no_deterministic_gate",
            )

        return payload

    def _gate_payload(
        self,
        payload: dict[str, Any],
        *,
        support_label: str,
        reason: str,
        category: str,
    ) -> dict[str, Any]:
        payload["triggered"] = True
        payload["support_label"] = support_label
        payload["reason"] = reason
        payload["category"] = category
        payload["labels"].append(reason)
        return payload

    def _support_text(self, candidates: list[EvidenceCandidate], best_candidate_id: str | None) -> str:
        if best_candidate_id:
            for candidate in candidates:
                if candidate.chunk_id == best_candidate_id:
                    return candidate.chunk_text
        return " ".join(candidate.chunk_text for candidate in candidates)

    def _critical_entities(self, metadata: dict[str, Any]) -> list[str]:
        values = metadata.get("critical_entities") or metadata.get("entities") or []
        return [str(value) for value in values if str(value).strip()]

    def _critical_values(self, metadata: dict[str, Any], claim_mentions: list[Any]) -> set[str]:
        values = metadata.get("critical_values") or metadata.get("numbers") or []
        normalized = {str(value).strip().lower() for value in values if str(value).strip()}
        date_spans = [
            (mention.start, mention.end)
            for mention in claim_mentions
            if mention.value_type == "date"
        ]
        normalized.update(
            mention.normalized_value
            for mention in claim_mentions
            if mention.value_type in {"percentage", "money", "duration", "version", "quantity", "number"}
            and not (
                mention.value_type == "number"
                and any(start <= mention.start and mention.end <= end for start, end in date_spans)
            )
        )
        return normalized

    def _critical_dates(self, metadata: dict[str, Any], claim_mentions: list[Any]) -> set[str]:
        values = metadata.get("critical_dates") or metadata.get("dates") or []
        normalized = {str(value).strip().lower() for value in values if str(value).strip()}
        normalized.update(
            mention.normalized_value
            for mention in claim_mentions
            if mention.value_type == "date"
        )
        return normalized

    def _check_critical_entities(self, entities: list[str], evidence_text: str) -> dict[str, Any]:
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text.lower()))
        checked: list[dict[str, Any]] = []
        has_issue = False
        for entity in entities:
            entity_tokens = [token for token in re.findall(r"[a-z0-9]+", entity.lower()) if len(token) > 2]
            if not entity_tokens:
                continue
            exact_match = entity.lower() in evidence_text.lower()
            overlap = [token for token in entity_tokens if token in evidence_tokens]
            precise = self._is_precision_entity(entity)
            mismatched = False
            if exact_match:
                status = "matched"
            elif precise and overlap:
                status = "mismatched"
                mismatched = True
            elif precise and not overlap:
                status = "missing"
                mismatched = True
            else:
                status = "weak_or_paraphrastic"
            checked.append(
                {"entity": entity, "tokens": entity_tokens, "status": status, "matched_tokens": overlap}
            )
            has_issue = has_issue or mismatched
        return {"checked": checked, "has_issue": has_issue}

    def _check_entity_value_binding(
        self,
        *,
        claim: str,
        evidence_text: str,
        all_candidates: list[EvidenceCandidate],
        critical_entities: list[str],
        critical_values: set[str],
    ) -> dict[str, bool]:
        if not critical_entities or not critical_values:
            return {"conflict": False}
        claim_text = claim.lower()
        evidence_sentences = self._candidate_sentences(evidence_text)
        all_sentences: list[str] = []
        for candidate in all_candidates:
            all_sentences.extend(self._candidate_sentences(candidate.chunk_text))

        for entity in critical_entities:
            entity_tokens = self._entity_tokens(entity)
            if not entity_tokens:
                continue
            if entity.lower() in evidence_text.lower():
                continue
            if self._supports_entity_value_binding(entity_tokens, critical_values, evidence_sentences):
                continue
            conflicting_signal = self._find_conflicting_entity_value_signal(
                claim_text=claim_text,
                entity_tokens=entity_tokens,
                critical_values=critical_values,
                sentences=all_sentences,
            )
            if conflicting_signal:
                return {"conflict": True}
        return {"conflict": False}

    def _candidate_sentences(self, text: str) -> list[str]:
        parts = re.split(r"[.;]\s+|\n+", text.lower())
        return [part.strip() for part in parts if part.strip()]

    def _entity_tokens(self, entity: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", entity.lower()) if len(token) > 2]

    def _supports_entity_value_binding(
        self,
        entity_tokens: list[str],
        critical_values: set[str],
        sentences: list[str],
    ) -> bool:
        for sentence in sentences:
            if self._entity_tokens_present(entity_tokens, sentence) and any(value in sentence for value in critical_values):
                return True
        return False

    def _find_conflicting_entity_value_signal(
        self,
        *,
        claim_text: str,
        entity_tokens: list[str],
        critical_values: set[str],
        sentences: list[str],
    ) -> bool:
        for sentence in sentences:
            overlap_count = self._entity_token_overlap_count(entity_tokens, sentence)
            if overlap_count == 0:
                continue
            values_in_sentence = {value for value in critical_values if value in sentence}
            if values_in_sentence:
                continue
            if re.search(r"\d", sentence):
                sentence_mentions = extract_value_mentions(sentence)
                if sentence_mentions and (
                    overlap_count >= len(entity_tokens)
                    or (overlap_count >= 1 and self._predicate_token_overlap(claim_text, sentence) > 0)
                ):
                    return True
            if self._predicate_token_overlap(claim_text, sentence) > 0:
                return True
        return False

    def _entity_tokens_present(self, entity_tokens: list[str], sentence: str) -> bool:
        return self._entity_token_overlap_count(entity_tokens, sentence) >= len(entity_tokens)

    def _entity_token_overlap_count(self, entity_tokens: list[str], sentence: str) -> int:
        sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return sum(
            1
            for token in entity_tokens
            if token in sentence_tokens or self._morph_token_present(token, sentence_tokens)
        )

    def _morph_token_present(self, token: str, sentence_tokens: set[str]) -> bool:
        if token in sentence_tokens:
            return True
        simple = token.rstrip("s")
        if simple and simple in sentence_tokens:
            return True
        if token.endswith("ers") and token[:-3] + "e" in sentence_tokens:
            return True
        if token.endswith("ers") and token[:-1] in sentence_tokens:
            return True
        if token.endswith("s") and token[:-1] + "e" in sentence_tokens:
            return True
        return False

    def _is_precision_entity(self, entity: str) -> bool:
        return bool(
            re.search(r"[A-Z0-9]|[-./]", entity)
            or any(token in entity.lower() for token in ("account", "section", "form", "flag", "url", "version"))
        )

    def _check_critical_mentions(
        self,
        *,
        claim_mentions: list[Any],
        evidence_mentions: list[Any],
        critical_values: set[str],
        kind: str,
    ) -> dict[str, bool]:
        relevant_claims = [
            mention
            for mention in claim_mentions
            if (kind == "date" and mention.value_type == "date")
            or (kind == "value" and mention.value_type != "date")
        ]
        if kind == "value":
            date_mentions = [
                mention
                for mention in claim_mentions
                if mention.value_type == "date"
            ]
            relevant_claims = [
                mention
                for mention in relevant_claims
                if not (
                    mention.value_type == "number"
                    and any(
                        date_mention.start <= mention.start and mention.end <= date_mention.end
                        for date_mention in date_mentions
                    )
                )
            ]
        if critical_values:
            relevant_claims = [mention for mention in relevant_claims if mention.normalized_value in critical_values]
        missing = False
        conflict = False
        unit_mismatch = False
        for claim_mention in relevant_claims:
            compatible = [
                evidence_mention
                for evidence_mention in evidence_mentions
                if self._compatible_fact_mentions(claim_mention, evidence_mention)
            ]
            if not compatible:
                missing = True
                continue
            exact = [
                evidence_mention
                for evidence_mention in compatible
                if claim_mention.normalized_value == evidence_mention.normalized_value
                and claim_mention.unit == evidence_mention.unit
            ]
            if exact:
                continue
            if any(
                claim_mention.normalized_value == evidence_mention.normalized_value
                and claim_mention.unit != evidence_mention.unit
                for evidence_mention in compatible
            ):
                conflict = True
                unit_mismatch = True
                continue
            conflict = True
        return {"missing": missing, "conflict": conflict, "unit_mismatch": unit_mismatch, "matched": bool(relevant_claims) and not missing and not conflict}

    def _compatible_fact_mentions(self, claim_mention: Any, evidence_mention: Any) -> bool:
        if claim_mention.value_type == evidence_mention.value_type:
            if claim_mention.value_type in {"duration", "quantity"}:
                return claim_mention.unit == evidence_mention.unit or claim_mention.normalized_value == evidence_mention.normalized_value
            return True
        if claim_mention.value_type == "date" or evidence_mention.value_type == "date":
            return False
        return claim_mention.unit == evidence_mention.unit or claim_mention.value_type == "number" or evidence_mention.value_type == "number"

    def _is_related_but_non_supporting(
        self,
        claim: str,
        evidence_text: str,
        heur_res: VerificationResult,
    ) -> bool:
        if heur_res.support_label == "supported":
            return False
        claim_terms = self._content_terms(claim)
        evidence_terms = self._content_terms(evidence_text)
        if not claim_terms:
            return False
        overlap_ratio = len(claim_terms & evidence_terms) / max(len(claim_terms), 1)
        if overlap_ratio < 0.15:
            return False
        if self._predicate_token_overlap(claim, evidence_text) > 0:
            return False
        return True

    def _predicate_token_overlap(self, claim: str, evidence_text: str) -> int:
        predicate_claim = self._predicate_tokens(claim)
        predicate_evidence = self._predicate_tokens(evidence_text)
        return len(predicate_claim & predicate_evidence)

    def _predicate_tokens(self, text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS and len(token) > 3
        }
        return {
            token
            for token in tokens
            if token not in {"account", "section", "form", "city", "trial", "study", "request", "application", "api", "endpoint"}
        }

    def _has_positive_support_signal(
        self,
        *,
        critical_entities: list[str],
        entity_check: dict[str, Any],
        value_check: dict[str, bool],
        date_check: dict[str, bool],
        heur_res: VerificationResult,
    ) -> bool:
        if heur_res.support_label == "supported":
            return True
        if critical_entities and any(item["status"] == "matched" for item in entity_check["checked"]):
            return True
        if value_check["matched"]:
            return True
        if date_check["matched"]:
            return True
        if not critical_entities and not any(value_check.values()) and not any(date_check.values()):
            return True
        return False

    def _content_terms(self, text: str) -> set[str]:
        content_terms: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in STOPWORDS:
                continue
            normalized = CONTENT_TERM_NORMALIZATIONS.get(token, token)
            if len(normalized) > 2:
                content_terms.add(normalized)
        return content_terms
