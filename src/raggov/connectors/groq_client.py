"""Minimal Groq client adapter for GovRAG LLM smoke tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


@dataclass
class GroqCallStats:
    call_count: int = 0
    rate_limited: bool = False


class GroqLLMClient:
    """Adapter exposing chat()/complete() for GovRAG claim components."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 20.0,
        json_mode: bool = True,
    ) -> None:
        from groq import Groq

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._json_mode = json_mode
        self._client = Groq(api_key=api_key)
        self.stats = GroqCallStats()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "groq"

    def chat(self, prompt: str) -> str:
        self.stats.call_count += 1
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._json_mode:
                try:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("response_format", None)
                    response = self._client.chat.completions.create(**fallback_kwargs)
                except Exception as fallback_exc:
                    self._capture_rate_limit(fallback_exc)
                    raise RuntimeError(self._safe_exception_text(fallback_exc)) from fallback_exc
            else:
                self._capture_rate_limit(exc)
                raise RuntimeError(self._safe_exception_text(exc)) from exc
        message = response.choices[0].message.content if response.choices else None
        return message or ""

    def complete(self, prompt: str) -> str:
        return self.chat(prompt)

    def _capture_rate_limit(self, exc: Exception) -> None:
        lowered = str(exc).lower()
        if "rate limit" in lowered or "429" in lowered:
            self.stats.rate_limited = True

    def _safe_exception_text(self, exc: Exception) -> str:
        return _redact_secret(str(exc), self._api_key)


def build_groq_client_from_env() -> tuple[GroqLLMClient | None, str | None]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "missing_api_key"
    model = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    try:
        client = GroqLLMClient(api_key=api_key, model=model, temperature=0.0, max_tokens=1024)
    except ImportError:
        return None, "missing_groq_package_install_via_pip_install_groq"
    except Exception as exc:
        return None, _redact_secret(str(exc), api_key)
    return client, None
