"""Google Gemini (Generative Language API) adapter, non-streaming."""

from __future__ import annotations

from typing import Optional, Sequence

import requests

from .base import ModelClient, ModelResponse


class GeminiClient(ModelClient):
    def _endpoint(self) -> str:
        base = (self.cfg.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        return f"{base}/models/{self.cfg.model}:generateContent"

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

        # Gemini parts: text first, then inline images.
        parts: list[dict] = [{"text": prompt}]
        for img in images or []:
            parts.append(
                {"inline_data": {"mime_type": "image/jpeg", "data": self.encode_image(img)}}
            )

        gen_config = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        resp = requests.post(
            self._endpoint(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.cfg.api_key},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        text, finish = "", None
        if candidates:
            cand = candidates[0]
            finish = cand.get("finishReason")
            for part in cand.get("content", {}).get("parts", []):
                if not part.get("thought"):
                    text += part.get("text", "")
        return ModelResponse(
            text=text,
            model=self.cfg.model,
            usage=data.get("usageMetadata", {}) or {},
            finish_reason=finish,
        )
