"""Decomposition layer for compound claims in GovRAG grounding."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from raggov.analyzers.grounding.claims import AtomicSubclaim, ExtractedClaim, _extract_entities, _extract_dates, _extract_numbers

logger = logging.getLogger(__name__)


class CompoundClaimDecomposer(ABC):
    """Interface for compound claim decomposers."""

    @abstractmethod
    def decompose(self, claim: ExtractedClaim) -> list[AtomicSubclaim]:
        """Decompose a compound claim into atomic subclaims."""
        raise NotImplementedError


class HeuristicConjunctionSplitter(CompoundClaimDecomposer):
    """
    A deterministic splitter that splits on conjunctions (and, but, while, whereas, etc.).
    Useful as a fast baseline and reliable fallback.
    """

    def decompose(self, claim: ExtractedClaim) -> list[AtomicSubclaim]:
        text = claim.claim_text.strip()
        # Split pattern: ", and " or " and " or " but " or " while " or " whereas "
        pattern = r"\s+(?:and|but|while|whereas)\s+|\s*,\s+(?:and|but|while|whereas)\s+"
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        
        # Clean parts
        parts = [p.strip() for p in parts if p.strip()]
        
        # If we didn't actually split it, just return the single part as subclaim
        if len(parts) <= 1:
            return [
                AtomicSubclaim(
                    parent_claim_id=claim.claim_id,
                    subclaim_id=f"{claim.claim_id}_sub_1",
                    text=text,
                    decomposition_method="heuristic_conjunction_split",
                    atomicity_status="atomic",
                    required_support=True,
                    entities=claim.entities,
                    dates=claim.dates,
                    numbers=claim.numbers,
                )
            ]

        subclaims = []
        for i, part in enumerate(parts, 1):
            # Ensure it ends with a period if the original did
            if text.endswith(".") and not part.endswith("."):
                part += "."
            
            # Simple heuristic: try to inherit entities, dates, numbers that are present in the part
            part_entities = [e for e in claim.entities if e.lower() in part.lower()]
            part_dates = [d for d in claim.dates if d.lower() in part.lower()]
            part_numbers = [n for n in claim.numbers if n.lower() in part.lower()]

            # Fall back to regex extraction if inherited list is empty
            if not part_entities:
                part_entities = _extract_entities(part)
            if not part_dates:
                part_dates = _extract_dates(part)
            if not part_numbers:
                part_numbers = _extract_numbers(part)

            subclaims.append(
                AtomicSubclaim(
                    parent_claim_id=claim.claim_id,
                    subclaim_id=f"{claim.claim_id}_sub_{i}",
                    text=part,
                    decomposition_method="heuristic_conjunction_split",
                    atomicity_status="atomic",
                    required_support=True,
                    entities=part_entities,
                    dates=part_dates,
                    numbers=part_numbers,
                )
            )
        return subclaims


class LLMCompoundClaimDecomposer(CompoundClaimDecomposer):
    """
    Decomposes compound claims using a configured LLM client.
    Falls back to HeuristicConjunctionSplitter if LLM call or parse fails.
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        self.fallback = HeuristicConjunctionSplitter()

    def decompose(self, claim: ExtractedClaim) -> list[AtomicSubclaim]:
        prompt = self._prompt(claim.claim_text)
        try:
            response = self._invoke(prompt)
            parsed = self._parse_response(response)
            
            subclaims = []
            for i, item in enumerate(parsed, 1):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                subclaims.append(
                    AtomicSubclaim(
                        parent_claim_id=claim.claim_id,
                        subclaim_id=f"{claim.claim_id}_sub_{i}",
                        text=text,
                        decomposition_method="llm_decomposer",
                        atomicity_status="atomic",
                        required_support=bool(item.get("required_support", True)),
                        entities=item.get("entities", _extract_entities(text)),
                        dates=item.get("dates", _extract_dates(text)),
                        numbers=item.get("numbers", _extract_numbers(text)),
                    )
                )
            if len(subclaims) >= 2:
                return subclaims
            
            logger.warning("LLM decomposer produced fewer than 2 subclaims. Falling back to heuristic.")
        except Exception as exc:
            logger.warning("LLM compound decomposition failed: %s. Falling back to heuristic.", exc)
            
        # Fallback to heuristic
        fallback_subs = self.fallback.decompose(claim)
        for sub in fallback_subs:
            sub.decomposition_method = "llm_fallback_to_heuristic"
        return fallback_subs

    def _invoke(self, prompt: str) -> str:
        client = self.llm_client
        if hasattr(client, "chat"):
            res = client.chat(prompt)
        elif hasattr(client, "complete"):
            res = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
        
        if isinstance(res, dict):
            if "text" in res:
                return res["text"]
            if "content" in res:
                return res["content"]
        return str(res)

    def _prompt(self, text: str) -> str:
        return (
            "Decompose the following compound claim into its constituent atomic subclaims.\n"
            "Each subclaim must be a single self-contained factual statement.\n"
            "Return only a strict JSON list of objects matching the schema below.\n\n"
            "Schema:\n"
            "[\n"
            "  {\n"
            '    "text": "string",\n'
            '    "required_support": true,\n'
            '    "entities": ["string"],\n'
            '    "dates": ["string"],\n'
            '    "numbers": ["string"]\n'
            "  }\n"
            "]\n\n"
            f"Compound Claim:\n{text}"
        )

    def _parse_response(self, response: str) -> list[dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "subclaims" in parsed:
            return parsed["subclaims"]
        raise ValueError("Invalid decomposer response format")


def build_decomposer(config: dict[str, Any]) -> CompoundClaimDecomposer:
    """Build decomposer based on config."""
    llm_client = config.get("llm_client")
    if llm_client is not None:
        return LLMCompoundClaimDecomposer(llm_client)
    return HeuristicConjunctionSplitter()
