"""Synchronous answer generation client for OpenAI-compatible chat endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from .prompting import build_prompt


class AnsweringClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=timeout)

    def answer(self, query: str, chunk_texts: list[str]) -> str:
        prompt = build_prompt(query, chunk_texts)

        try:
            response = self._http_client.post(
                self._base_url,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Answering request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text.strip()
            message = f"Answering request failed with status {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Answering response was not valid JSON") from exc

        return self._parse_content(payload)

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    def _parse_content(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Answering response must be a JSON object")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Answering response missing choices[0].message.content")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Answering response missing choices[0].message.content")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Answering response missing choices[0].message.content")

        content = message.get("content")
        if isinstance(content, str) and content:
            return content

        raise RuntimeError("Answering response missing choices[0].message.content")

    def __enter__(self) -> AnsweringClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
