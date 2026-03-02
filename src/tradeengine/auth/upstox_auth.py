from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


class UpstoxAuthError(RuntimeError):
    """Raised when authentication with Upstox fails."""


@dataclass(frozen=True)
class UpstoxCredentials:
    api_key: str
    api_secret: str
    redirect_uri: str


class UpstoxAuth:
    AUTH_BASE_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

    def __init__(
        self,
        credentials: UpstoxCredentials,
        timeout_seconds: int = 15,
        session: requests.Session | None = None,
    ) -> None:
        self._credentials = credentials
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._access_token: str | None = None
        self._expires_at_utc: datetime | None = None

    def generate_login_url(self, state: str | None = None) -> str:
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self._credentials.api_key,
            "redirect_uri": self._credentials.redirect_uri,
        }
        if state:
            params["state"] = state
        return f"{self.AUTH_BASE_URL}?{urlencode(params)}"

    def exchange_auth_code_for_access_token(self, auth_code: str) -> str:
        payload = {
            "code": auth_code,
            "client_id": self._credentials.api_key,
            "client_secret": self._credentials.api_secret,
            "redirect_uri": self._credentials.redirect_uri,
            "grant_type": "authorization_code",
        }

        try:
            response = self._session.post(
                self.TOKEN_URL,
                data=payload,
                timeout=self._timeout_seconds,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            logger.exception("Token exchange request failed")
            raise UpstoxAuthError("Unable to reach Upstox token endpoint") from exc

        if response.status_code >= 400:
            logger.error(
                "Token exchange failed with status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise UpstoxAuthError("Token exchange failed due to invalid credentials or request")

        try:
            data = response.json()
        except ValueError as exc:
            logger.exception("Non-JSON token response received")
            raise UpstoxAuthError("Token endpoint returned malformed response") from exc

        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            logger.error("Token response missing access_token field: %s", data)
            raise UpstoxAuthError("Token response missing access token")

        expires_in = data.get("expires_in")
        self.set_access_token(token=token, expires_in_seconds=expires_in)
        return token

    def set_access_token(self, token: str, expires_in_seconds: Any | None = None) -> None:
        self._access_token = token
        self._expires_at_utc = None
        if isinstance(expires_in_seconds, int) and expires_in_seconds > 0:
            self._expires_at_utc = datetime.now(UTC) + timedelta(seconds=expires_in_seconds)

    @property
    def access_token(self) -> str:
        if self._access_token is None:
            raise UpstoxAuthError("Access token not available")

        if self._expires_at_utc is not None and datetime.now(UTC) >= self._expires_at_utc:
            raise UpstoxAuthError("Access token expired")

        return self._access_token
