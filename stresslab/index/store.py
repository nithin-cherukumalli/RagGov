"""Small local vector index backed by NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    payload: Any


class VectorIndex:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._payloads: list[Any] = []
        self._vectors: list[np.ndarray] = []
        self._dimension: int | None = None

    def add(self, chunk_id: str, vector: list[float], payload: Any) -> None:
        array = self._normalize_vector(vector)
        if self._dimension is None:
            self._dimension = int(array.shape[0])
        elif int(array.shape[0]) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {int(array.shape[0])}"
            )

        self._chunk_ids.append(chunk_id)
        self._vectors.append(array)
        self._payloads.append(payload)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0 or not self._vectors:
            return []

        query = self._normalize_vector(query_vector)
        if self._dimension is not None and int(query.shape[0]) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {int(query.shape[0])}"
            )

        matrix = np.vstack(self._vectors)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(query))

        if query_norm == 0.0:
            similarities = np.zeros(len(self._vectors), dtype=np.float64)
        else:
            dots = matrix @ query
            denom = matrix_norms * query_norm
            similarities = np.divide(
                dots,
                denom,
                out=np.zeros_like(dots, dtype=np.float64),
                where=denom != 0.0,
            )

        limit = min(top_k, len(self._vectors))
        ranked = np.argsort(-similarities, kind="stable")[:limit]
        return [
            SearchResult(
                chunk_id=self._chunk_ids[index],
                score=float(similarities[index]),
                payload=self._payloads[index],
            )
            for index in ranked
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors = (
            np.vstack(self._vectors)
            if self._vectors
            else np.empty((0, self._dimension or 0), dtype=np.float64)
        )
        np.savez_compressed(
            target,
            chunk_ids=np.array(self._chunk_ids, dtype=object),
            payloads=np.array(self._payloads, dtype=object),
            vectors=vectors,
        )

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        archive = np.load(Path(path), allow_pickle=True)
        index = cls()
        chunk_ids = archive["chunk_ids"].tolist()
        payloads = archive["payloads"].tolist()
        vectors = archive["vectors"]

        for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads, strict=True):
            index.add(str(chunk_id), vector.tolist(), payload)
        return index

    @staticmethod
    def _normalize_vector(vector: list[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("Vectors must be one-dimensional")
        return array
