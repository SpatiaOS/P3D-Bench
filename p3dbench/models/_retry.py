"""Shared HTTP POST with retry/backoff for the model adapters.

Ports cadbenchmark's ``_post_with_retry`` policy: exponential backoff on HTTP
429 / 5xx and transient transport errors (Timeout / ConnectionError /
ChunkedEncodingError), so a flaky provider no longer turns one network hiccup
into a failed eval case. The budget defaults to 2 total attempts (1 retry),
overridable via ``P3DBENCH_API_MAX_RETRIES`` (mirrors ``CADBENCHMARK_API_MAX_RETRIES``).

Some OpenAI-compatible relays (notably OpenRouter) answer **HTTP 200 with a
provider-error body** — e.g. ``{"error": {"code": 522, ...}}`` and no
``choices`` — instead of a 5xx. ``retryable_body`` lets an adapter fold those
into the same backoff rather than surfacing them as an un-retried parse error.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

RETRY_DELAY_S = 5


def api_max_retries() -> int:
    """Total attempts per call (1 = no retry). Default 2, env-overridable."""
    try:
        return max(1, int(os.getenv("P3DBENCH_API_MAX_RETRIES", "2")))
    except ValueError:
        return 2


def post_with_retry(
    url: str,
    *,
    headers: dict,
    json_body: dict,
    timeout: int,
    max_retries: Optional[int] = None,
    retryable_body: Optional[Callable[[dict], Optional[str]]] = None,
) -> requests.Response:
    """POST with exponential backoff on 429 / 5xx and transient transport errors.

    ``retryable_body(parsed_json) -> reason | None`` optionally flags an HTTP-200
    provider-error body as retryable. Raises the last error (or ``RuntimeError``)
    once the budget is exhausted; the caller treats that as the case's error state.
    """
    if max_retries is None:
        max_retries = api_max_retries()
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if resp.ok:
                reason = None
                if retryable_body is not None:
                    try:
                        reason = retryable_body(resp.json())
                    except Exception:
                        reason = None
                if reason:
                    if attempt < max_retries - 1:
                        wait = RETRY_DELAY_S * (2 ** attempt)
                        logger.warning(
                            "Retryable provider error in 200 body (%s), waiting %ss "
                            "(attempt %d/%d)...", reason, wait, attempt + 1, max_retries)
                        time.sleep(wait)
                        continue
                    raise requests.RequestException(f"provider error in 200 body: {reason}")
                return resp

            try:
                logger.error("Error details: %s", resp.json())
            except Exception:
                logger.error("Error response: %s", resp.text[:500])

            is_retryable = resp.status_code == 429 or resp.status_code >= 500
            if is_retryable and attempt < max_retries - 1:
                wait = RETRY_DELAY_S * (2 ** attempt)
                logger.warning("Retryable error %s, waiting %ss (attempt %d/%d)...",
                               resp.status_code, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise requests.RequestException(f"API request failed: HTTP {resp.status_code}")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = RETRY_DELAY_S * (2 ** attempt)
                logger.warning("Connection error %s, retrying in %ss (attempt %d/%d)...",
                               type(e).__name__, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"POST failed after {max_retries} retries") from last_exc


def openrouter_error_reason(data: dict) -> Optional[str]:
    """Reason string when an OpenAI-compatible HTTP-200 body is a provider error."""
    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
        err = data["error"]
        if isinstance(err, dict):
            return f"code={err.get('code')} {err.get('message')}"
        return str(err)
    return None
