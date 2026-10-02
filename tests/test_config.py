import pytest

from agentops.config import Settings
from agentops.errors import ConfigError

ENV_NAMES = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_EXTRA_MODELS",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "DATABASE_URL",
    "ENABLE_CODE_MODE",
    "DAILY_RUN_CAP",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests must not depend on the developer's real environment or .env file."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def make_settings() -> Settings:
    return Settings(_env_file=None)


def test_defaults_are_safe():
    settings = make_settings()
    assert settings.enable_code_mode is False
    assert settings.gemini_api_key is None
    assert settings.daily_run_cap == 30


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "abc123")
    assert make_settings().require("tavily_api_key") == "abc123"


def test_secrets_never_appear_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    settings = make_settings()
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)


def test_require_raises_clear_error_when_missing():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        make_settings().require("gemini_api_key")


def test_require_treats_empty_string_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ConfigError):
        make_settings().require("gemini_api_key")


def test_require_works_for_plain_string_settings(monkeypatch):
    with pytest.raises(ConfigError, match="GEMINI_MODEL"):
        make_settings().require("gemini_model")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    assert make_settings().require("gemini_model") == "some-model"


def test_extra_gemini_models_default_to_empty_and_read_from_environment(monkeypatch):
    assert make_settings().gemini_extra_models == ""
    monkeypatch.setenv("GEMINI_EXTRA_MODELS", "model-b, model-c")
    assert make_settings().gemini_extra_models == "model-b, model-c"
