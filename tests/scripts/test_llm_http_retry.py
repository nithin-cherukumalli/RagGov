"""Unit tests for the LLM HTTP backoff helper (offline; no real network)."""

from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _llm_http import post_json_with_retry  # noqa: E402


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example/x",
        code=code,
        msg="err",
        hdrs=headers or {},
        fp=io.BytesIO(b"{}"),
    )


def _make_urlopen(sequence: list[object]):
    """Return a fake urlopen that yields/raises items from ``sequence`` in order."""
    calls = {"n": 0}

    def _urlopen(_request, timeout=None):  # noqa: ANN001
        item = sequence[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    return _urlopen, calls


def test_retries_then_succeeds_on_429():
    slept: list[float] = []
    urlopen, calls = _make_urlopen(
        [_http_error(429), _http_error(429), _FakeResp(b'{"ok": true}')]
    )
    out = post_json_with_retry(
        "https://example/x",
        {"p": 1},
        {},
        timeout=5,
        urlopen=urlopen,
        sleep=slept.append,
    )
    assert out == {"ok": True}
    assert calls["n"] == 3
    assert len(slept) == 2  # slept before each retry


def test_does_not_retry_client_error_401():
    slept: list[float] = []
    urlopen, calls = _make_urlopen([_http_error(401)])
    with pytest.raises(urllib.error.HTTPError):
        post_json_with_retry(
            "https://example/x", {}, {}, timeout=5, urlopen=urlopen, sleep=slept.append
        )
    assert calls["n"] == 1  # no retry on auth error
    assert slept == []


def test_exhausts_retries_then_raises_original():
    slept: list[float] = []
    urlopen, calls = _make_urlopen([_http_error(503)] * 10)
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json_with_retry(
            "https://example/x",
            {},
            {},
            timeout=5,
            max_retries=3,
            urlopen=urlopen,
            sleep=slept.append,
        )
    assert exc.value.code == 503
    assert calls["n"] == 4  # 1 initial + 3 retries
    assert len(slept) == 3


def test_honors_numeric_retry_after():
    slept: list[float] = []
    urlopen, _ = _make_urlopen(
        [_http_error(429, {"Retry-After": "7"}), _FakeResp(b"{}")]
    )
    post_json_with_retry(
        "https://example/x",
        {},
        {},
        timeout=5,
        base_delay=0.0,  # isolate Retry-After from jitter/backoff
        urlopen=urlopen,
        sleep=slept.append,
    )
    assert slept and slept[0] == pytest.approx(7.0)
