import json
import importlib
import subprocess

import httpx
import pytest

from stresslab.answering import AnsweringClient, build_prompt

answering_module = importlib.import_module("stresslab.answering.client")


def test_build_prompt_includes_query_context_and_citation_requirement() -> None:
    prompt = build_prompt(
        "What is Rule 5?",
        ["[doc-1] Rule 5 requires notice.", "[doc-2] Appendix details the process."],
    )

    assert "What is Rule 5?" in prompt
    assert "[doc-1] Rule 5 requires notice." in prompt
    assert "[doc-2] Appendix details the process." in prompt
    assert "cite" in prompt.lower()


def test_answering_client_posts_chat_completion_request_and_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("http://example.com/v1/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == "answer-model"
        assert payload["messages"] == [
            {
                "role": "user",
                "content": build_prompt(
                    "What is Rule 5?",
                    ["[doc-1] Rule 5 requires notice."],
                ),
            }
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Rule 5 requires notice. [doc-1]"}}]},
        )

    client = AnsweringClient(
        base_url="http://example.com/v1/chat/completions",
        model="answer-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    answer = client.answer("What is Rule 5?", ["[doc-1] Rule 5 requires notice."])

    assert answer == "Rule 5 requires notice. [doc-1]"


def test_answering_client_raises_clear_error_for_invalid_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = AnsweringClient(
        base_url="http://example.com/v1/chat/completions",
        model="answer-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Answering response missing choices\\[0\\]\\.message\\.content"):
        client.answer("What is Rule 5?", ["chunk text"])


def test_answering_client_falls_back_to_curl_when_python_networking_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenClient:
        def post(self, *args, **kwargs):
            raise httpx.ConnectError("host is down")

    def fake_run(cmd, capture_output, text, check):
        assert "http://example.com/v1/chat/completions" in cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Fallback answer. [doc-1]"
                            }
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(answering_module.subprocess, "run", fake_run)

    client = AnsweringClient(
        base_url="http://example.com/v1/chat/completions",
        model="answer-model",
        http_client=BrokenClient(),
    )

    answer = client.answer("What is Rule 5?", ["[doc-1] Rule 5 requires notice."])

    assert answer == "Fallback answer. [doc-1]"
