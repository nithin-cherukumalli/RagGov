"""Minimal Groq LLM client adapter for RagGov's optional LLM/NLI tier.

Conforms to the `llm_client` contract the verifiers expect: a `.chat(prompt) -> str`
method (see LLMClaimEntailmentVerifierV1._invoke). OpenAI-compatible endpoint, stdlib
only (urllib), no extra deps.

SECURITY: the API key is read from the GROQ_API_KEY environment variable. It is NEVER
read from a file or hardcoded here. Do not commit a key.

Sandbox note: api.groq.com is proxy-blocked inside the RagGov build sandbox (403). Run
this on a machine with open network access (e.g. your laptop):

    GROQ_API_KEY=... PYTHONPATH=src:. python scripts/run_nli_heldout.py
"""

from __future__ import annotations

import json
import os
import urllib.request

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """Tiny OpenAI-compatible chat client for Groq."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._key = os.environ.get("GROQ_API_KEY")
        if not self._key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it in the environment; never hardcode it."
            )

    def chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    # Some verifier paths probe for .complete(); alias to chat.
    def complete(self, prompt: str) -> str:
        return self.chat(prompt)
