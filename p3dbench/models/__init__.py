"""Model client factory: provider name -> client class."""

from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT_CONFIG_DIR, ModelConfig, get_model_config
from .anthropic import AnthropicClient
from .base import ModelClient, ModelResponse
from .gemini import GeminiClient
from .openai_compatible import OpenAICompatibleClient

_PROVIDERS = {
    "openai_compatible": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def build_client(cfg: ModelConfig) -> ModelClient:
    if cfg.provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{cfg.provider}' for model '{cfg.name}'. "
            f"Choices: {', '.join(sorted(_PROVIDERS))}"
        )
    return _PROVIDERS[cfg.provider](cfg)


def get_client(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> ModelClient:
    return build_client(get_model_config(name, config_dir))


__all__ = [
    "ModelClient",
    "ModelResponse",
    "build_client",
    "get_client",
]
