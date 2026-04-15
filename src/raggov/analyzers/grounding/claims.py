"""Utilities for extracting claims from generated answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


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

    def _extract_deterministic(self, answer: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:\s+|$)", answer)
            if sentence.strip()
        ]
        return [sentence for sentence in sentences if len(sentence.split()) >= 6]

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
