"""Configuration loading.

BYOK design: ``configs/models.yaml`` holds only metadata (provider, model id,
base URL, and the *name* of the env var that holds the key). Secrets live in
the environment / ``.env`` — never in YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT_S = 600

# Aliases accepted in models.yaml `provider:` — all OpenAI-compatible
# endpoints (OpenRouter, vLLM, LM Studio, ...) share one adapter.
PROVIDER_ALIASES = {
    "openai": "openai_compatible",
    "openrouter": "openai_compatible",
}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader (KEY=VALUE lines). Existing env vars are not overridden."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: int = DEFAULT_TIMEOUT_S
    extra: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Model '{self.name}' needs the env var {self.api_key_env} "
                f"(see .env.example); it is empty or unset."
            )
        return key


@dataclass
class JudgeConfig:
    judge_model: str
    decompose_model: str
    n_views: int = 4
    judge_timeout_s: int = 240
    decompose_timeout_s: int = 300


def _config_path(config_dir: Path, filename: str) -> Path:
    path = Path(config_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run from the repo root or pass --config-dir."
        )
    return path


def load_models_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(_config_path(config_dir, "models.yaml").read_text(encoding="utf-8"))
    models: dict[str, ModelConfig] = {}
    for name, block in (raw.get("models") or {}).items():
        block = dict(block)
        provider = str(block.pop("provider", "openai_compatible")).lower()
        provider = PROVIDER_ALIASES.get(provider, provider)
        models[name] = ModelConfig(
            name=name,
            provider=provider,
            model=str(block.pop("model", name)),
            api_key_env=str(block.pop("api_key_env", "")),
            base_url=block.pop("base_url", None),
            temperature=float(block.pop("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(block.pop("max_tokens", DEFAULT_MAX_TOKENS)),
            timeout_s=int(block.pop("timeout_s", DEFAULT_TIMEOUT_S)),
            extra=block,
        )
    return models


def get_model_config(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> ModelConfig:
    models = load_models_config(config_dir)
    if name not in models:
        raise KeyError(
            f"Model '{name}' is not registered in {Path(config_dir) / 'models.yaml'}. "
            f"Available: {', '.join(sorted(models))}"
        )
    return models[name]


def load_judge_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> JudgeConfig:
    raw = yaml.safe_load(_config_path(config_dir, "judge.yaml").read_text(encoding="utf-8"))
    return JudgeConfig(
        judge_model=raw["judge_model"],
        decompose_model=raw["decompose_model"],
        n_views=int(raw.get("n_views", 4)),
        judge_timeout_s=int(raw.get("judge_timeout_s", 240)),
        decompose_timeout_s=int(raw.get("decompose_timeout_s", 300)),
    )
