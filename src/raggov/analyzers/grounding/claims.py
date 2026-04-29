"""Utilities for extracting claims from generated answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

# Matches dotted abbreviation chains like "G.O.Rt.No." that must not be sentence-split.
_ABBREV_CHAIN_RE = re.compile(r"^[A-Z](?:\.[A-Z0-9]\w*)+\.$")

_SUBSTANTIVE_RE = re.compile(
    r"\d"  # any digit: numbers, dates, amounts, percentages, GO numbers
    r"|\$|€|£|₹|%"  # currency/percentage symbols
    r"|G\.O\b"  # Government Order prefix
    r"|\b(?:must|shall|required?|mandated?|prohibit(?:ed)?|exempt|approv(?:al|ed)|"
    r"authoriz(?:ed)?|comply|compliance|permit(?:ted)?|regulations?|rule|act|policy|"
    r"circular|notification|order|deadline|threshold|applicable|effective|enforce(?:d)?|"
    r"mandatory|optional|waive(?:r)?|gazette)\b",
    re.IGNORECASE,
)


class ClaimExtractor:
    """Split generated answers into atomic factual claims."""

    def __init__(self, use_llm: bool = False, llm_client: object | None = None) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client

    def extract(self, answer: str) -> list[str]:
        """Extract claim strings from an answer."""
        if self.use_llm and self.llm_client is not None:
            try:
                return self._extract_with_llm(answer)
            except Exception as exc:
                logger.warning("LLM claim extraction failed, falling back to deterministic: %s", exc)
                return self._extract_deterministic(answer)
        return self._extract_deterministic(answer)

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

    def _extract_deterministic(self, answer: str) -> list[str]:
        raw = [
            s.strip()
            for s in re.split(r"(?<=[.!?])(?:\s+|$)", answer)
            if s.strip()
        ]
        sentences = self._merge_abbreviation_splits(raw)
        return [s for s in sentences if len(s.split()) >= 6 or self._is_substantive(s)]

    def _extract_with_llm(self, answer: str) -> list[str]:
        prompt = (
            "Split the following answer into atomic, self-contained factual claims. "
            "Return only a JSON array of strings, one claim per element. "
            f"Answer: {answer}"
        )
        client = self.llm_client
        if hasattr(client, "chat"):
            response = client.chat(prompt)  # type: ignore[union-attr]
        elif hasattr(client, "complete"):
            response = client.complete(prompt)  # type: ignore[union-attr]
        else:
            raise TypeError("llm_client must provide chat() or complete()")

        parsed = self._parse_response(response)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("claim extractor response must be a JSON array of strings")
        return [claim.strip() for claim in parsed if claim.strip()]

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
        if not isinstance(response, str):
            response = str(response)
        return json.loads(response)
