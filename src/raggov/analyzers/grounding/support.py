"""Analyzer for assessing claim support against retrieved evidence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal


logger = logging.getLogger(__name__)

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.grounding.value_extraction import find_value_alignment
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
)
from raggov.models.run import RAGRun


NEGATION_SIGNALS = {"not", "never", "no", "no longer", "contrary to"}
ANCHOR_WEIGHT = 0.6
CONTENT_TERM_NORMALIZATIONS = {
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
REMEDIATION = (
    "{failed} of {total} claims are unsupported by retrieved context. "
    "Review retrieval quality or add source verification."
)


class ClaimGroundingAnalyzer(BaseAnalyzer):
    """Assess whether generated claims are grounded in retrieved chunks."""

    weight = 0.9

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        claim_extractor_client = self.config.get("claim_extractor_client")
        extractor = ClaimExtractor(
            use_llm=claim_extractor_client is not None,
            llm_client=claim_extractor_client,
        )
        claims = extractor.extract(run.final_answer)
        if not claims:
            return self.skip("no claims extracted from final answer")

        claim_results = [self._evaluate_claim(claim, run.query, run.retrieved_chunks) for claim in claims]
        failed_results = [
            result
            for result in claim_results
            if result.label in {"unsupported", "contradicted"}
        ]
        contradicted_results = [
            result for result in claim_results if result.label == "contradicted"
        ]
        failed_fraction = len(failed_results) / len(claim_results)
        entailed_count = sum(1 for result in claim_results if result.label == "entailed")
        unsupported_count = sum(1 for result in claim_results if result.label == "unsupported")
        contradicted_count = sum(1 for result in claim_results if result.label == "contradicted")
        evidence = [
            (
                "Claim grounding summary: "
                f"total={len(claim_results)}, "
                f"entailed={entailed_count}, "
                f"unsupported={unsupported_count}, "
                f"contradicted={contradicted_count}"
            )
        ]
        fallback_count = sum(1 for result in claim_results if result.fallback_used)
        evidence.append(
            "Claim verification methods: "
            f"structured={len(claim_results) - fallback_count}, "
            f"fallback={fallback_count}"
        )
        remediation = REMEDIATION.format(
            failed=len(failed_results),
            total=len(claim_results),
        )

        if failed_fraction >= float(self.config.get("fail_threshold", 0.3)):
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )
        if contradicted_results:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.CONTRADICTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )
        if failed_results:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GROUNDING,
                evidence=evidence,
                claim_results=claim_results,
                remediation=remediation,
            )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=evidence,
            claim_results=claim_results,
        )

    def _evaluate_claim(
        self, claim: str, query: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        if bool(self.config.get("use_llm", False)) and self.config.get("llm_client") is not None:
            try:
                return self._evaluate_claim_with_llm(claim, chunks)
            except Exception as exc:
                logger.warning("LLM claim grounding failed, falling back to deterministic: %s", exc)
                fallback = self._evaluate_claim_deterministic(claim, chunks)
                fallback.fallback_used = True
                fallback.evidence_reason = (
                    "LLM verifier unavailable; fell back to deterministic overlap/anchor check"
                )
                return fallback

        try:
            return self._evaluate_claim_structured(claim, query, chunks)
        except Exception as exc:
            logger.warning("Structured claim verifier failed, falling back to deterministic: %s", exc)
            fallback = self._evaluate_claim_deterministic(claim, chunks)
            fallback.fallback_used = True
            fallback.evidence_reason = (
                "Structured verifier unavailable; fell back to deterministic overlap/anchor check"
            )
            return fallback

    def _evaluate_claim_structured(
        self, claim: str, query: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        """Structured evidence-aware verification using best-evidence chunk analysis."""
        if bool(self.config.get("force_structured_verifier_error", False)):
            raise RuntimeError("forced structured verifier failure")
        if not chunks:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[],
                contradicting_chunk_ids=[],
                confidence=0.0,
                verification_method="structured_claim_verifier_v1",
                evidence_reason="No retrieved chunks available for evidence matching.",
                calibration_status="uncalibrated",
            )

        claim_terms = self._content_terms(claim)
        claim_anchors = self._extract_anchors(claim)
        if not claim_terms:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[],
                contradicting_chunk_ids=[],
                confidence=0.0,
                verification_method="structured_claim_verifier_v1",
                evidence_reason="Claim has no meaningful content terms to verify.",
                calibration_status="uncalibrated",
            )

        best_chunk, best_score = self._best_evidence_chunk(claim_terms, claim_anchors, chunks)
        if best_chunk is None:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[],
                contradicting_chunk_ids=[],
                confidence=0.0,
                verification_method="structured_claim_verifier_v1",
                evidence_reason="No candidate evidence chunk matched claim content terms.",
                calibration_status="uncalibrated",
            )

        claim_has_numeric_anchor = any(char.isdigit() for anchor in claim_anchors for char in anchor)
        chunk_anchor_set = set(self._extract_anchors(best_chunk.text))
        claim_anchor_set = set(claim_anchors)
        anchor_hits = len(claim_anchor_set & chunk_anchor_set) if claim_anchor_set else 0
        numeric_anchor_mismatch = bool(claim_anchor_set) and claim_has_numeric_anchor and anchor_hits == 0

        if self._contains_negation_of_terms(best_chunk.text, claim_terms):
            return ClaimResult(
                claim_text=claim,
                label="contradicted",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
                confidence=best_score,
                verification_method="value_aware_structured_claim_verifier_v1",
                evidence_reason=(
                    "Best evidence chunk contains negation near claim terms, indicating contradiction."
                ),
                calibration_status="uncalibrated",
            )

        value_matches, value_conflicts, missing_critical_value = find_value_alignment(
            claim,
            best_chunk.text,
        )
        if value_conflicts:
            conflict = value_conflicts[0]
            return ClaimResult(
                claim_text=claim,
                label="contradicted",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
                confidence=best_score,
                verification_method="value_aware_structured_claim_verifier_v1",
                evidence_reason=(
                    f"Claim states {conflict['claim_value']}, but evidence states {conflict['evidence_value']} "
                    f"for the same {conflict['value_type']} context."
                ),
                calibration_status="uncalibrated",
                value_conflicts=value_conflicts,
                value_matches=value_matches,
            )
        if missing_critical_value:
            claim_value = _first_claim_value_snippet(claim)
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[],
                confidence=best_score,
                verification_method="value_aware_structured_claim_verifier_v1",
                evidence_reason=(
                    f"Claim value {claim_value} is not present in related evidence."
                    if claim_value
                    else "Claim contains critical value details not present in related evidence."
                ),
                calibration_status="uncalibrated",
                value_conflicts=value_conflicts,
                value_matches=value_matches,
            )

        if numeric_anchor_mismatch and not value_matches:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[],
                confidence=best_score,
                verification_method="value_aware_structured_claim_verifier_v1",
                evidence_reason=(
                    "High lexical overlap but factual anchors (numbers/dates/entities) do not match."
                ),
                calibration_status="uncalibrated",
                value_conflicts=value_conflicts,
                value_matches=value_matches,
            )

        label: Literal["entailed", "unsupported", "contradicted"] = (
            "entailed"
            if (best_score >= 0.5 or (value_matches and best_score >= 0.2))
            else "unsupported"
        )
        reason = (
            "Claim supported by best evidence chunk via content-term and anchor match."
            if label == "entailed"
            else "Best evidence chunk does not provide enough content-term/anchor support."
        )

        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=[best_chunk.chunk_id] if (label == "entailed" and best_score >= 0.2) else [],
            candidate_chunk_ids=[best_chunk.chunk_id],
            contradicting_chunk_ids=[],
            confidence=best_score,
            verification_method="value_aware_structured_claim_verifier_v1",
            evidence_reason=reason,
            calibration_status="uncalibrated",
            value_conflicts=value_conflicts,
            value_matches=value_matches,
        )

    def _evaluate_claim_deterministic(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        """Score grounding via normalized content-term coverage plus factual anchors.

        This is still deterministic, but it is less brittle than raw token overlap:
        content terms approximate the claim predicate, while anchors (numbers, dates,
        and likely named entities) are weighted more heavily because factual support
        usually depends on matching those verifiable anchors.
        """
        claim_terms = self._content_terms(claim)
        claim_anchors = self._extract_anchors(claim)
        if not claim_terms:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[],
                contradicting_chunk_ids=[],
                confidence=0.0,
                verification_method="deterministic_overlap_anchor_v0",
                evidence_reason="Claim has no meaningful content terms after normalization.",
                calibration_status="uncalibrated",
                fallback_used=False,
            )

        best_chunk: RetrievedChunk | None = None
        best_score = 0.0
        term_weight = 1.0 - float(self.config.get("anchor_weight", ANCHOR_WEIGHT))
        anchor_weight = float(self.config.get("anchor_weight", ANCHOR_WEIGHT))
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
            return ClaimResult(
                claim_text=claim,
                label="contradicted",
                supporting_chunk_ids=[],
                candidate_chunk_ids=[best_chunk.chunk_id],
                contradicting_chunk_ids=[best_chunk.chunk_id],
                confidence=best_score,
                verification_method="deterministic_overlap_anchor_v0",
                evidence_reason="Negation detected near overlapping claim terms.",
                calibration_status="uncalibrated",
                fallback_used=False,
            )

        if best_chunk is not None:
            value_matches, value_conflicts, missing_critical_value = find_value_alignment(
                claim,
                best_chunk.text,
            )
            if value_conflicts:
                conflict = value_conflicts[0]
                return ClaimResult(
                    claim_text=claim,
                    label="contradicted",
                    supporting_chunk_ids=[],
                    candidate_chunk_ids=[best_chunk.chunk_id],
                    contradicting_chunk_ids=[best_chunk.chunk_id],
                    confidence=best_score,
                    verification_method="deterministic_overlap_anchor_v0",
                    evidence_reason=(
                        f"Claim states {conflict['claim_value']}, but evidence states {conflict['evidence_value']} "
                        f"for the same {conflict['value_type']} context."
                    ),
                    calibration_status="uncalibrated",
                    fallback_used=False,
                    value_conflicts=value_conflicts,
                    value_matches=value_matches,
                )
            if missing_critical_value:
                claim_value = _first_claim_value_snippet(claim)
                return ClaimResult(
                    claim_text=claim,
                    label="unsupported",
                    supporting_chunk_ids=[],
                    candidate_chunk_ids=[best_chunk.chunk_id],
                    contradicting_chunk_ids=[],
                    confidence=best_score,
                    verification_method="deterministic_overlap_anchor_v0",
                    evidence_reason=(
                        f"Claim value {claim_value} is not present in related evidence."
                        if claim_value
                        else "Claim contains critical value details not present in related evidence."
                    ),
                    calibration_status="uncalibrated",
                    fallback_used=False,
                    value_conflicts=value_conflicts,
                    value_matches=value_matches,
                )
        else:
            value_matches = []
            value_conflicts = []

        supporting_chunk_ids = [best_chunk.chunk_id] if best_chunk is not None else []
        candidate_chunk_ids = [best_chunk.chunk_id] if best_chunk is not None else []
        if best_score >= 0.5 or (value_matches and best_score >= 0.2):
            label: Literal["entailed", "unsupported", "contradicted"] = "entailed"
        else:
            label = "unsupported"

        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=(
                supporting_chunk_ids if (label == "entailed" and best_score >= 0.2) else []
            ),
            candidate_chunk_ids=candidate_chunk_ids if best_chunk is not None else [],
            contradicting_chunk_ids=[],
            confidence=best_score,
            verification_method="deterministic_overlap_anchor_v0",
            evidence_reason=(
                "Best chunk crosses support threshold via overlap and anchor scoring."
                if label == "entailed"
                else "Best chunk score below support threshold."
            ),
            calibration_status="uncalibrated",
            fallback_used=False,
            value_conflicts=value_conflicts,
            value_matches=value_matches,
        )

    def _evaluate_claim_with_llm(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        payload = self._call_llm(claim, chunks)
        label = payload.get("label")
        if label not in {"entailed", "unsupported", "contradicted"}:
            raise ValueError("invalid LLM grounding label")
        evidence_chunk_id = payload.get("evidence_chunk_id")
        supporting_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id else []
        candidate_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id else []
        contradicting_chunk_ids = (
            [str(evidence_chunk_id)] if evidence_chunk_id and label == "contradicted" else []
        )
        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=supporting_chunk_ids if label == "entailed" else [],
            candidate_chunk_ids=candidate_chunk_ids,
            contradicting_chunk_ids=contradicting_chunk_ids,
            confidence=float(payload.get("confidence", 0.0)),
            verification_method="llm_claim_verifier_v1",
            evidence_reason=str(payload.get("rationale", "LLM verifier output")),
            calibration_status="uncalibrated",
            fallback_used=False,
        )

    def _call_llm(self, claim: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        client = self.config["llm_client"]
        prompt = self._prompt(claim, chunks)
        if hasattr(client, "chat"):
            response = client.chat(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
        parsed = self._parse_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM grounding response must be a JSON object")
        return parsed

    def _prompt(self, claim: str, chunks: list[RetrievedChunk]) -> str:
        relevant_chunks = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks
        )
        return (
            "Does the following retrieved context support, contradict, or neither "
            "support nor contradict this claim?\n"
            f"Context: {relevant_chunks}\n"
            f"Claim: {claim}\n"
            'Answer with JSON: {"label": "entailed"|"unsupported"|"contradicted", '
            '"confidence": 0.0-1.0, "evidence_chunk_id": "chunk_id or null", '
            '"rationale": "short reason"}'
        )

    def _best_evidence_chunk(
        self, claim_terms: set[str], claim_anchors: list[str], chunks: list[RetrievedChunk]
    ) -> tuple[RetrievedChunk | None, float]:
        best_chunk: RetrievedChunk | None = None
        best_score = 0.0
        term_weight = 1.0 - float(self.config.get("anchor_weight", ANCHOR_WEIGHT))
        anchor_weight = float(self.config.get("anchor_weight", ANCHOR_WEIGHT))

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

    def _contains_negation_of_terms(self, text: str, terms: set[str]) -> bool:
        tokens = self._tokens(text)
        for index, token in enumerate(tokens):
            normalized_token = self._normalize_content_term(token)
            window = tokens[max(0, index - 5) : index + 6]
            window_text = " ".join(window)
            if normalized_token in terms and any(
                re.search(rf"\b{re.escape(signal)}\b", window_text)
                for signal in NEGATION_SIGNALS
            ):
                return True
        return False

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in self._tokens(text)
            if token not in STOPWORDS
        }

    def _content_terms(self, text: str) -> set[str]:
        """Return normalized content-bearing terms for deterministic grounding."""
        content_terms = set()
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
        """Normalize light morphological variants used in factual paraphrases."""
        if not token:
            return ""
        normalized = CONTENT_TERM_NORMALIZATIONS.get(token, token)
        if normalized.endswith("ies") and len(normalized) > 4:
            return normalized[:-3] + "y"
        if normalized.endswith("s") and len(normalized) > 4 and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized

    def _extract_anchors(self, text: str) -> list[str]:
        """Extract verifiable anchors such as numbers, dates, and likely named entities."""
        anchors: list[str] = []
        lowered = text.lower()

        anchors.extend(
            match.group(0)
            for match in re.finditer(r"(?:[$€£])?\d[\d,]*(?:\.\d+)?%?", lowered)
        )

        for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            value = match.group(0)
            if " " in value or match.start() > 0:
                anchors.append(value.lower())

        for match in re.finditer(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b", text):
            anchors.append(match.group(0).lower())

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


def _first_claim_value_snippet(claim_text: str) -> str | None:
    match = re.search(r"(?:[$₹€£]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?%?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))*\b)",
        claim_text.lower(),
    )
    return match.group(0) if match else None
