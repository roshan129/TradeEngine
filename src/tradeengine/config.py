from __future__ import annotations

from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing."""


@dataclass(frozen=True)
class UpstoxConfig:
    api_key: str
    api_secret: str
    redirect_uri: str


class Settings(BaseSettings):
    app_name: str = "TradeEngine"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    upstox_api_key: str | None = None
    upstox_api_secret: str | None = None
    upstox_redirect_uri: str | None = None
    upstox_access_token: str | None = None
    upstox_auth_code: str | None = None
    upstox_instrument_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


def get_upstox_config() -> UpstoxConfig:
    missing: list[str] = []
    if not settings.upstox_api_key:
        missing.append("UPSTOX_API_KEY")
    if not settings.upstox_api_secret:
        missing.append("UPSTOX_API_SECRET")
    if not settings.upstox_redirect_uri:
        missing.append("UPSTOX_REDIRECT_URI")

    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    return UpstoxConfig(
        api_key=settings.upstox_api_key,
        api_secret=settings.upstox_api_secret,
        redirect_uri=settings.upstox_redirect_uri,
    )
