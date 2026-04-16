from pathlib import Path

import pytest

from stresslab.index import SearchResult, VectorIndex


def test_search_returns_highest_cosine_matches_first():
    index = VectorIndex()
    index.add("alpha", [1.0, 0.0], {"text": "alpha"})
    index.add("beta", [0.8, 0.2], {"text": "beta"})
    index.add("gamma", [0.0, 1.0], {"text": "gamma"})

    results = index.search([1.0, 0.0], top_k=2)

    assert results == [
        SearchResult(chunk_id="alpha", score=pytest.approx(1.0), payload={"text": "alpha"}),
        SearchResult(
            chunk_id="beta",
            score=pytest.approx(0.9701425, rel=1e-6),
            payload={"text": "beta"},
        ),
    ]


def test_search_returns_empty_results_for_empty_index():
    index = VectorIndex()

    assert index.search([1.0, 0.0], top_k=3) == []


def test_add_rejects_dimension_mismatch():
    index = VectorIndex()
    index.add("alpha", [1.0, 0.0], {"text": "alpha"})

    with pytest.raises(ValueError, match="Vector dimension mismatch: expected 2, got 3"):
        index.add("beta", [1.0, 0.0, 0.0], {"text": "beta"})


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "index.npz"
    index = VectorIndex()
    index.add("alpha", [1.0, 0.0], {"rank": 1})
    index.add("beta", [0.0, 1.0], {"rank": 2})
    index.save(path)

    loaded = VectorIndex.load(path)

    assert loaded.search([0.0, 1.0], top_k=2) == [
        SearchResult(chunk_id="beta", score=pytest.approx(1.0), payload={"rank": 2}),
        SearchResult(chunk_id="alpha", score=pytest.approx(0.0), payload={"rank": 1}),
    ]
