from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Secrets: SecretStr keeps them out of repr(), logs and tracebacks.
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    serper_api_key: SecretStr | None = None
    database_url: SecretStr | None = None

    # Model names change often, so they are configuration, never hard-coded.
    gemini_model: str = ""
    groq_model: str = ""

    log_level: str = "INFO"
    enable_code_mode: bool = False  # jailed Python analysis: OFF unless explicitly enabled
    daily_run_cap: int = Field(default=30, gt=0)

    def require(self, field: str) -> str:
        """Return a setting's plain value, or raise a clear error if it is unset/empty."""
        raw = getattr(self, field)
        value = raw.get_secret_value() if isinstance(raw, SecretStr) else raw
        if not value:
            raise ConfigError(f"{field.upper()} is not set. Add it to your .env file.")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
