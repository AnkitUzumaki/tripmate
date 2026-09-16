"""Application configuration. The single source of truth for every tunable."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together_ai": "TOGETHER_AI_API_KEY",
}

KEYLESS_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})

DEFAULT_PROVIDER = "openai"


class ConfigError(Exception):
    """Raised at startup when configuration is unusable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_s: int = Field(default=30, gt=0)

    # --- RAG ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_path: str = "./.chroma"
    destinations_dir: str = "./data/destinations"
    rag_top_k: int = Field(default=4, gt=0)
    rag_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Agent ---
    max_tool_iterations: int = Field(default=5, gt=0)
    max_query_chars: int = Field(default=2000, gt=0)
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # --- Weather ---
    weather_timeout_s: float = Field(default=3.0, gt=0)
    weather_cache_dir: str = "./.weather_cache"
    weather_climate_years: int = Field(default=5, gt=0, le=20)

    # --- Persistence ---
    database_url: str = "sqlite:///./tripmate.db"

    # --- Observability ---
    log_level: str = "INFO"
    log_format: str = "rich"
    trace_dir: str = "./traces"

    @property
    def provider(self) -> str:
        """Provider slug taken from the model string, e.g. 'anthropic/claude' -> 'anthropic'."""
        if "/" not in self.llm_model:
            return DEFAULT_PROVIDER
        return self.llm_model.split("/", 1)[0]

    def export_provider_key(self) -> None:
        """Copy `llm_api_key` into the env var LiteLLM expects for this provider.

        LiteLLM reads provider-specific variables (OPENAI_API_KEY, GROQ_API_KEY, ...).
        Users configure one `LLM_API_KEY`; this maps it to the right place at startup.
        """
        provider = self.provider
        if provider in KEYLESS_PROVIDERS:
            return

        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var is None:
            supported = ", ".join(sorted(PROVIDER_ENV_VARS) + sorted(KEYLESS_PROVIDERS))
            raise ConfigError(
                f"Unknown provider {provider!r} in LLM_MODEL={self.llm_model!r}. "
                f"Supported: {supported}"
            )

        if not self.llm_api_key:
            raise ConfigError(
                f"LLM_API_KEY is required for provider {provider!r}. "
                f"Set it in .env, or switch LLM_MODEL to an ollama/* model."
            )

        os.environ[env_var] = self.llm_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
