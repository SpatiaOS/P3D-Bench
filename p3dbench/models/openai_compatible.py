"""OpenAI-compatible chat/completions adapter.

Covers OpenAI, OpenRouter, vLLM, LM Studio — anything that speaks the
``/v1/chat/completions`` protocol. Single non-streaming POST.
"""

from __future__ import annotations

from typing import Optional, Sequence

import requests

from .base import IMAGE_MIME, ModelClient, ModelResponse


class OpenAICompatibleClient(ModelClient):
    def _endpoint(self) -> str:
        base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

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

        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images or []:
            b64 = self.encode_image(img)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME};base64,{b64}"}}
            )

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload = {"model": self.cfg.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature

        resp = requests.post(
            self._endpoint(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        text = message.get("content") or ""
        if isinstance(text, list):  # some routers return content as a parts list
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return ModelResponse(
            text=text,
            model=data.get("model", self.cfg.model),
            usage=data.get("usage", {}) or {},
            finish_reason=choice.get("finish_reason"),
        )
