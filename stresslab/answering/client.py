"""Synchronous answer generation client for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import subprocess
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
        self._timeout = timeout
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
            try:
                payload = self._curl_post(
                    {
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
            except RuntimeError:
                raise RuntimeError(f"Answering request failed: {exc}") from exc
            return self._parse_content(payload)

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

    def _curl_post(self, payload: dict[str, Any]) -> Any:
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "--max-time",
                str(int(self._timeout)),
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
                self._base_url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Answering curl fallback failed: {detail}")
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("Answering response was not valid JSON") from exc

    def __enter__(self) -> AnsweringClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
