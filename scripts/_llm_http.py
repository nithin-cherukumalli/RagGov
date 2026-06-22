"""Shared HTTP-with-retry helper for the optional LLM/NLI clients (stdlib only).

The hybrid NLI tier was measured at ~42% claim-check fallback on the real heldout — almost
all of it transient rate-limiting (HTTP 429) and the occasional 5xx. Every fallback degrades
the NLI signal the grounded-clean gate depends on. This helper adds bounded exponential
backoff with jitter so a rate-limited call waits and retries instead of immediately dropping
to the heuristic verifier. It is the reliability half of "make NLI runs reliable + cheap".

Design notes:
- Retries only idempotent-safe transient statuses (429 + 5xx) and transient URL errors.
- Honors a numeric ``Retry-After`` header when present (Groq/Moonshot both send one).
- ``urlopen`` and ``sleep`` are injectable so the backoff logic is unit-testable offline
  (the build sandbox proxy-blocks the real endpoints).
- On exhaustion it re-raises the ORIGINAL error so callers keep surfacing real messages.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable

# Transient statuses worth retrying. 4xx other than 429 are caller/auth errors and must NOT
# be retried (retrying a 401/400 just wastes the quota and hides the real bug).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Return the Retry-After delay in seconds, if the server sent a numeric one."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None  # HTTP-date form is not worth parsing for our short-lived retries
    return seconds if seconds >= 0 else None


def post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed response, retrying transient failures.

    Raises the original ``HTTPError`` / ``URLError`` once retries are exhausted so callers can
    format provider-specific error messages exactly as before.
    """
    jitter = (rng or random).uniform
    attempt = 0
    while True:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt >= max_retries:
                raise
            retry_after = _parse_retry_after(exc)
            backoff = min(max_delay, base_delay * (2 ** attempt))
            delay = retry_after if retry_after is not None else backoff
            sleep(delay + jitter(0.0, base_delay))
            attempt += 1
        except urllib.error.URLError:
            # Transient connection error (DNS hiccup, reset). Retry a bounded number of times.
            if attempt >= max_retries:
                raise
            sleep(min(max_delay, base_delay * (2 ** attempt)) + jitter(0.0, base_delay))
            attempt += 1
