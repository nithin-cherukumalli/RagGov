"""Minimal Kimi (Moonshot AI) LLM client for RagGov's optional LLM/NLI tier.

Kimi is OpenAI-compatible (https://api.moonshot.ai/v1). Conforms to the `llm_client`
contract the verifiers expect: `.chat(prompt) -> str` (and `.complete`). Stdlib only.

SECURITY: the key is read from the environment (KIMI_API_KEY / MOONSHOT_API_KEY) or a
gitignored `.env` line like `KIMI_API_KEY = sk-...`. NEVER hardcoded or committed.

Config via env (all optional):
  KIMI_API_KEY     your Moonshot/Kimi key (or MOONSHOT_API_KEY, or `kimi_api` in .env)
  KIMI_MODEL       default model id (e.g. kimi-k2.5, moonshot-v1-8k)
  KIMI_BASE_URL    default https://api.moonshot.ai/v1  (use .cn region if needed)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _llm_http import post_json_with_retry  # noqa: E402

_KEY_NAMES = ("KIMI_API_KEY", "MOONSHOT_API_KEY", "kimi_api", "moonshot_api")
DEFAULT_KIMI_MODEL = "kimi-k2.5"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


def _load_kimi_key() -> str | None:
    for name in _KEY_NAMES:
        val = os.environ.get(name)
        if val:
            return val.strip().strip("'\"")
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


class KimiClient:
    """OpenAI-compatible chat client for Kimi / Moonshot."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
        *,
        max_retries: int = 5,
    ) -> None:
        self.model = model or os.environ.get("KIMI_MODEL") or DEFAULT_KIMI_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        if self.model.startswith("kimi-"):
            self.temperature = 1.0
        self.base_url = (base_url or os.environ.get("KIMI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._key = _load_kimi_key()
        if not self._key:
            raise RuntimeError(
                "No Kimi key found. Set KIMI_API_KEY in the environment or add "
                "`KIMI_API_KEY = sk-...` to a gitignored .env file. Never hardcode it."
            )

    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        # Moonshot requires temperature in (0, 1]; temperature=0 is rejected. Clamp.
        temperature = self.temperature if self.temperature and self.temperature > 0 else 0.3
        if temperature > 1:
            temperature = 1.0
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            data = post_json_with_retry(
                f"{self.base_url}/chat/completions",
                payload,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                timeout=90,
                max_retries=self.max_retries,
            )
        except urllib.error.HTTPError as exc:  # surface Moonshot's real error message
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise RuntimeError(f"Kimi HTTP {exc.code} ({self.model}): {body[:400]}") from exc
        return data["choices"][0]["message"]["content"]

    def complete(self, prompt: str) -> str:
        return self.chat(prompt)

    def list_models(self) -> list[str]:
        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self._key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))
