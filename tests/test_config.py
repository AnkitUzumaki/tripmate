import pytest

from tripmate.config import ConfigError, Settings


def test_provider_parsed_from_prefixed_model():
    settings = Settings(llm_model="anthropic/claude-haiku-4-5", llm_api_key="k")
    assert settings.provider == "anthropic"


def test_provider_defaults_to_openai_when_model_is_unprefixed():
    settings = Settings(llm_model="gpt-4o-mini", llm_api_key="k")
    assert settings.provider == "openai"


def test_export_provider_key_writes_provider_specific_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(llm_model="groq/llama-3.3-70b-versatile", llm_api_key="secret")

    settings.export_provider_key()

    import os
    assert os.environ["GROQ_API_KEY"] == "secret"


def test_export_provider_key_is_a_noop_for_keyless_ollama(monkeypatch):
    settings = Settings(llm_model="ollama/qwen3.5", llm_api_key=None)

    settings.export_provider_key()  # must not raise


def test_export_provider_key_raises_when_required_key_is_missing():
    settings = Settings(llm_model="gpt-4o-mini", llm_api_key=None)

    with pytest.raises(ConfigError, match="LLM_API_KEY is required"):
        settings.export_provider_key()


def test_export_provider_key_raises_on_unknown_provider():
    settings = Settings(llm_model="madeup/model-x", llm_api_key="k")

    with pytest.raises(ConfigError, match="Unknown provider"):
        settings.export_provider_key()


def test_query_char_limit_must_be_positive():
    with pytest.raises(ValueError):
        Settings(llm_api_key="k", max_query_chars=0)


def test_export_provider_key_uses_the_env_var_litellm_actually_reads_for_together_ai(monkeypatch):
    monkeypatch.delenv("TOGETHERAI_API_KEY", raising=False)
    settings = Settings(llm_model="together_ai/mixtral", llm_api_key="secret")

    settings.export_provider_key()

    import os
    assert os.environ["TOGETHERAI_API_KEY"] == "secret"
