import json
import importlib
import subprocess

import httpx
import pytest

from stresslab.embeddings import EmbeddingClient

embedding_module = importlib.import_module("stresslab.embeddings.client")


def test_embed_texts_returns_vectors_in_input_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://example.com/v1/embeddings")
        assert request.method == "POST"
        assert json.loads(request.content) == {
            "input": ["alpha", "beta"],
            "model": "test-model",
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    client = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.embed_texts(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_texts_uses_cache_for_repeated_calls():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.5, 0.25]}]},
        )

    client = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="cache-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    first = client.embed_texts(["repeat-me"])
    second = client.embed_texts(["repeat-me"])

    assert first == [[0.5, 0.25]]
    assert second == [[0.5, 0.25]]
    assert calls == 1


def test_embed_texts_cache_is_scoped_by_model():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0]}]},
        )

    transport = httpx.MockTransport(handler)
    first = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="model-a",
        http_client=httpx.Client(transport=transport),
    )
    second = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="model-b",
        http_client=httpx.Client(transport=transport),
    )

    first.embed_texts(["same-text"])
    second.embed_texts(["same-text"])

    assert requests == [
        {"input": ["same-text"], "model": "model-a"},
        {"input": ["same-text"], "model": "model-b"},
    ]


def test_embed_texts_raises_clear_error_for_http_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    client = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Embedding request failed with status 503"):
        client.embed_texts(["alpha"])


def test_embed_texts_raises_clear_error_for_invalid_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0}]})

    client = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Embedding response missing data\\[0\\]\\.embedding"):
        client.embed_texts(["alpha"])


def test_embed_texts_falls_back_to_curl_when_python_networking_fails(monkeypatch: pytest.MonkeyPatch):
    class BrokenClient:
        def post(self, *args, **kwargs):
            raise httpx.ConnectTimeout("timed out")

    def fake_run(cmd, capture_output, text, check):
        assert "http://example.com/v1/embeddings" in cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(embedding_module.subprocess, "run", fake_run)

    client = EmbeddingClient(
        base_url="http://example.com/v1/embeddings",
        model="curl-model",
        http_client=BrokenClient(),
    )

    assert client.embed_texts(["alpha"]) == [[0.1, 0.2, 0.3]]
