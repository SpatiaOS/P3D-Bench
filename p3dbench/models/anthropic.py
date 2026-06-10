"""Anthropic Messages API adapter (native, non-streaming)."""

from __future__ import annotations

from typing import Optional, Sequence

import requests

from .base import ModelClient, ModelResponse

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(ModelClient):
    def _endpoint(self) -> str:
        base = (self.cfg.base_url or "https://api.anthropic.com/v1").rstrip("/")
        return base if base.endswith("/messages") else f"{base}/messages"

    def generate(
        self,
        prompt: str,
        *,
        images: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> ModelResponse:
        if not prompt.strip():
            raise ValueError("Empty prompt")
        temperature, max_tokens, timeout = self._resolve(temperature, max_tokens, timeout)

        # Anthropic vision: image blocks first, then text.
        content: list[dict] = []
        for img in images or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": self.encode_image(img),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        resp = requests.post(
            self._endpoint(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.cfg.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return ModelResponse(
            text=text,
            model=data.get("model", self.cfg.model),
            usage=data.get("usage", {}) or {},
            finish_reason=data.get("stop_reason"),
        )
