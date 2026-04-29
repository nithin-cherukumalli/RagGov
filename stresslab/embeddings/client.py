"""Synchronous embedding client with a small in-memory cache."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from typing import Any

import httpx


class EmbeddingClient:
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
        self._cache: dict[str, list[float]] = {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        uncached_positions: list[int] = []
        uncached_texts: list[str] = []
        vectors: list[list[float] | None] = [None] * len(texts)

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = self._cache.get(cache_key)
            if cached is None:
                uncached_positions.append(index)
                uncached_texts.append(text)
            else:
                vectors[index] = list(cached)

        if uncached_texts:
            response = self._request_embeddings(uncached_texts)
            for position, text, vector in zip(
                uncached_positions, uncached_texts, response, strict=True
            ):
                self._cache[self._cache_key(text)] = list(vector)
                vectors[position] = list(vector)

        return [vector for vector in vectors if vector is not None]

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._http_client.post(
                self._base_url,
                json={"input": texts, "model": self._model},
            )
        except httpx.HTTPError as exc:
            try:
                payload = self._curl_post({"input": texts, "model": self._model})
            except RuntimeError:
                raise RuntimeError(f"Embedding request failed: {exc}") from exc
            return self._parse_embeddings_payload(payload, texts)

        if response.status_code >= 400:
            detail = response.text.strip()
            message = f"Embedding request failed with status {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Embedding response was not valid JSON") from exc

        return self._parse_embeddings_payload(payload, texts)

    def _parse_embeddings_payload(
        self,
        payload: Any,
        texts: list[str],
    ) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response missing data list")

        embeddings_by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("Embedding response data items must be objects")
            item_index = item.get("index")
            if not isinstance(item_index, int):
                raise RuntimeError("Embedding response missing integer data[index].index")
            embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError(
                    f"Embedding response missing data[{item_index}].embedding"
                )
            if not isinstance(embedding, list) or not all(
                isinstance(value, (int, float)) for value in embedding
            ):
                raise RuntimeError(
                    f"Embedding response data[{item_index}].embedding must be a list of numbers"
                )
            embeddings_by_index[item_index] = [float(value) for value in embedding]

        if len(embeddings_by_index) != len(texts):
            raise RuntimeError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings_by_index)}"
            )

        try:
            return [embeddings_by_index[index] for index in range(len(texts))]
        except KeyError as exc:
            raise RuntimeError(
                f"Embedding response missing data[{exc.args[0]}].embedding"
            ) from exc

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
            raise RuntimeError(f"Embedding curl fallback failed: {detail}")
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("Embedding response was not valid JSON") from exc

    def _cache_key(self, text: str) -> str:
        return f"{self._model}:{sha256(text.encode('utf-8')).hexdigest()}"

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
