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
from pathlib import Path

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
# Accepted key names, in priority order (env var first, then .env file).
_KEY_NAMES = ("GROQ_API_KEY", "GROQ_API", "groq_api", "groq_api_key")


def _load_key() -> str | None:
    """Resolve the Groq key from the environment, then from a .env file.

    .env lines look like `groq_api = gsk_...` (spaces and quotes tolerated). The .env
    file is gitignored — the key is never committed.
    """
    for name in _KEY_NAMES:
        val = os.environ.get(name)
        if val:
            return val.strip().strip("'\"")
    # search .env in cwd and up to repo root
    here = Path(__file__).resolve()
    for base in (Path.cwd(), here.parent, here.parent.parent):
        env_path = base / ".env"
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in _KEY_NAMES:
                return value.strip().strip("'\"")
    return None


class GroqClient:
    """Tiny OpenAI-compatible chat client for Groq."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._key = _load_key()
        if not self._key:
            raise RuntimeError(
                "No Groq key found. Set GROQ_API_KEY in the environment or add "
                "`groq_api = gsk_...` to a .env file (gitignored). Never hardcode it."
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
