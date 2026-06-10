"""Provider-agnostic model client.

Single-shot by design: one prompt in, one response out. No retry loops,
no error-feedback refinement, no relay routing.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..config import ModelConfig

logger = logging.getLogger(__name__)

# All providers receive images the same way: JPEG, longest edge capped.
IMAGE_MAX_EDGE = 1536
IMAGE_JPEG_QUALITY = 85
IMAGE_MIME = "image/jpeg"


@dataclass
class ModelResponse:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None


class ModelClient(ABC):
    """One client per registered model. ``generate`` performs exactly one call."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @abstractmethod
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
        """Send one request. ``images`` is a list of local file paths."""

    # -- shared helpers -------------------------------------------------

    def _resolve(self, temperature, max_tokens, timeout):
        return (
            self.cfg.temperature if temperature is None else temperature,
            self.cfg.max_tokens if max_tokens is None else max_tokens,
            self.cfg.timeout_s if timeout is None else timeout,
        )

    @staticmethod
    def encode_image(image_path: str) -> str:
        """File -> base64 JPEG (RGB, longest edge <= 1536 px, quality 85)."""
        data = Path(image_path).read_bytes()
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            if max(img.size) > IMAGE_MAX_EDGE:
                img.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
            data = buf.getvalue()
        except ImportError:
            logger.debug("Pillow not installed; sending image bytes as-is")
        return base64.b64encode(data).decode("utf-8")
