"""Utilities for extracting claims from generated answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Matches dotted abbreviation chains like "G.O.Rt.No." that must not be sentence-split.
_ABBREV_CHAIN_RE = re.compile(r"^[A-Z](?:\.[A-Z0-9]\w*)+\.$")

_SUBSTANTIVE_RE = re.compile(
    r"\d"  # any digit: numbers, dates, amounts, percentages, GO numbers
    r"|\$|€|£|₹|%"  # currency/percentage symbols
    r"|G\.O\b"  # Government Order prefix
    r"|\b(?:must|shall|require(?:s|d)?|mandated?|prohibit(?:ed)?|exempt|approv(?:al|ed)|"
    r"authoriz(?:ed)?|comply|compliance|permit(?:ted)?|regulations?|rules?|act|polic(?:y|ies)|"
    r"circular|notification|order|deadline|threshold|applicable|effective|enforce(?:d)?|"
    r"mandatory|optional|waive(?:r)?|gazette)\b",
    re.IGNORECASE,
)


class ExtractedClaim(BaseModel):
    """Structured factual claim extracted from a generated answer."""

    claim_id: str
    claim_text: str
    source_sentence: str
    source_start_char: int
    source_end_char: int
    atomicity_status: Literal["atomic", "compound", "unclear"]
    claim_type: str
    extraction_method: str
    extraction_reason: str
    should_verify: bool
    skip_reason: str | None = None


class ClaimExtractor:
    """Split generated answers into atomic factual claims."""

    def __init__(self, use_llm: bool = False, llm_client: object | None = None) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client

    def extract(self, answer: str) -> list[str]:
        """Extract claim strings from an answer (backward compatibility)."""
        claims = self.extract_structured(answer)
        return [c.claim_text for c in claims if c.should_verify]

    def extract_structured(self, answer: str) -> list[ExtractedClaim]:
        """Extract structured factual claims from an answer."""
        if self.use_llm and self.llm_client is not None:
            try:
                return self._extract_structured_llm(answer)
            except Exception as exc:
                logger.warning("LLM claim extraction failed, falling back to deterministic: %s", exc)
                fallback_claims = self._extract_structured_deterministic(answer)
                for c in fallback_claims:
                    c.extraction_method = "llm_fallback"
                return fallback_claims
        return self._extract_structured_deterministic(answer)

    @staticmethod
    def _is_substantive(sentence: str) -> bool:
        return _SUBSTANTIVE_RE.search(sentence) is not None

    @staticmethod
    def _merge_abbreviation_splits(fragments: list[str]) -> list[str]:
        merged: list[str] = []
        i = 0
        while i < len(fragments):
            if i + 1 < len(fragments) and _ABBREV_CHAIN_RE.match(fragments[i]):
                merged.append(fragments[i] + " " + fragments[i + 1])
                i += 2
            else:
                merged.append(fragments[i])
                i += 1
        return merged

    def _extract_structured_deterministic(self, answer: str) -> list[ExtractedClaim]:
        raw = [
            s.strip()
            for s in re.split(r"(?<=[.!?])(?:\s+|$)", answer)
            if s.strip()
        ]
        sentences = self._merge_abbreviation_splits(raw)
        
        claims: list[ExtractedClaim] = []
        search_start = 0
        
        for i, sentence in enumerate(sentences):
            start_char = answer.find(sentence, search_start)
            if start_char == -1:
                start_char = search_start
            end_char = start_char + len(sentence)
            search_start = end_char
            
            substantive_matches = len(_SUBSTANTIVE_RE.findall(sentence))
            conjunctions = len(re.findall(r"\b(?:and|or|but)\b", sentence, re.IGNORECASE))
            
            atomicity: Literal["atomic", "compound", "unclear"] = (
                "compound" if conjunctions > 0 or substantive_matches > 1 else "atomic"
            )
            
            is_substantive = substantive_matches > 0
            is_long_enough = len(sentence.split()) >= 6
            
            should_verify = True
            skip_reason = None
            if not is_substantive and not is_long_enough:
                should_verify = False
                skip_reason = "short_non_substantive"
            elif not is_substantive:
                should_verify = False
                skip_reason = "lacks_substantive_terms"
                
            claims.append(ExtractedClaim(
                claim_id=f"claim_{i}",
                claim_text=sentence,
                source_sentence=sentence,
                source_start_char=start_char,
                source_end_char=end_char,
                atomicity_status=atomicity,
                claim_type="statement",
                extraction_method="deterministic_v0_heuristic",
                extraction_reason="heuristic_sentence_split",
                should_verify=should_verify,
                skip_reason=skip_reason,
            ))
            
        return claims

    def _extract_structured_llm(self, answer: str) -> list[ExtractedClaim]:
        import uuid
        
        prompt = (
            "Split the following answer into atomic, self-contained factual claims. "
            "Return only a JSON array of objects, one object per claim. "
            "Each object MUST contain the following fields:\n"
            "- claim_text (string): The atomic factual claim.\n"
            "- source_sentence (string): The original sentence it was derived from.\n"
            "- atomicity_status (string): Either 'atomic' or 'compound'.\n"
            "- should_verify (boolean): true if it contains facts, false if purely rhetorical.\n"
            "Ensure you split compound statements, preserve dates/amounts/GO numbers, "
            "and do not invent facts. "
            f"\n\nAnswer: {answer}"
        )
        client = self.llm_client
        if hasattr(client, "chat"):
            response = client.chat(prompt)  # type: ignore[union-attr]
        elif hasattr(client, "complete"):
            response = client.complete(prompt)  # type: ignore[union-attr]
        else:
            raise TypeError("llm_client must provide chat() or complete()")

        parsed = self._parse_response(response)
        if not isinstance(parsed, list):
            raise ValueError("claim extractor response must be a JSON array")
        
        claims: list[ExtractedClaim] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            
            claim_text = item.get("claim_text", "")
            if not claim_text:
                continue
                
            claims.append(ExtractedClaim(
                claim_id=item.get("claim_id", f"claim_{uuid.uuid4().hex[:8]}"),
                claim_text=claim_text,
                source_sentence=item.get("source_sentence", claim_text),
                source_start_char=item.get("source_start_char", 0),
                source_end_char=item.get("source_end_char", 0),
                atomicity_status=item.get("atomicity_status", "atomic"),  # type: ignore
                claim_type=item.get("claim_type", "statement"),
                extraction_method=item.get("extraction_method", "llm"),
                extraction_reason=item.get("extraction_reason", "llm_extracted"),
                should_verify=item.get("should_verify", True),
                skip_reason=item.get("skip_reason", None),
            ))
            
        return claims

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
        if not isinstance(response, str):
            response = str(response)
        return json.loads(response)
