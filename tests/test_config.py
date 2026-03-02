from __future__ import annotations

import pytest

from tradeengine.config import ConfigError, get_upstox_config, settings


def test_get_upstox_config_fails_when_required_env_missing() -> None:
    prev_key = settings.upstox_api_key
    prev_secret = settings.upstox_api_secret
    prev_uri = settings.upstox_redirect_uri

    settings.upstox_api_key = None
    settings.upstox_api_secret = None
    settings.upstox_redirect_uri = None

    try:
        with pytest.raises(ConfigError):
            get_upstox_config()
    finally:
        settings.upstox_api_key = prev_key
        settings.upstox_api_secret = prev_secret
        settings.upstox_redirect_uri = prev_uri
