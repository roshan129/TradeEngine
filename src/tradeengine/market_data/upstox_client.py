from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests

from tradeengine.auth.upstox_auth import UpstoxAuth, UpstoxAuthError

logger = logging.getLogger(__name__)


class UpstoxClientError(RuntimeError):
    """Raised for Upstox market data client errors."""


class UpstoxClient:
    def __init__(
        self,
        auth: UpstoxAuth,
        base_url: str = "https://api.upstox.com/v3",
        timeout_seconds: int = 15,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._session = session or requests.Session()

    def fetch_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_datetime: datetime,
        to_datetime: datetime,
    ) -> dict[str, Any]:
        try:
            token = self._auth.access_token
        except UpstoxAuthError as exc:
            raise UpstoxClientError("Unable to call market data API without valid token") from exc

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        unit, interval_value = self._parse_interval(interval)
        to_date = to_datetime.date().isoformat()
        from_date = from_datetime.date().isoformat()
        path = f"/historical-candle/{instrument_key}/{unit}/{interval_value}/{to_date}/{from_date}"

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    f"{self._base_url}{path}",
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            except requests.Timeout as exc:
                logger.warning("Upstox candle request timeout on attempt %s/%s", attempt, self._max_retries)
                if attempt == self._max_retries:
                    raise UpstoxClientError("Timed out while fetching historical candles") from exc
                self._backoff(attempt)
                continue
            except requests.RequestException as exc:
                logger.warning(
                    "Upstox candle request network failure on attempt %s/%s",
                    attempt,
                    self._max_retries,
                )
                if attempt == self._max_retries:
                    raise UpstoxClientError("Network error while fetching historical candles") from exc
                self._backoff(attempt)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "Temporary Upstox error status=%s attempt=%s/%s",
                    response.status_code,
                    attempt,
                    self._max_retries,
                )
                if attempt == self._max_retries:
                    raise UpstoxClientError(
                        f"Temporary API failure persisted (status={response.status_code})"
                    )
                self._backoff(attempt)
                continue

            if 400 <= response.status_code < 500:
                logger.error(
                    "Non-retryable Upstox client error status=%s body=%s",
                    response.status_code,
                    response.text,
                )
                raise UpstoxClientError(f"Upstox rejected request (status={response.status_code})")

            if response.status_code >= 500:
                logger.error(
                    "Upstox server error status=%s body=%s", response.status_code, response.text
                )
                raise UpstoxClientError(f"Upstox server error (status={response.status_code})")

            try:
                return response.json()
            except ValueError as exc:
                logger.exception("Received malformed JSON from Upstox historical endpoint")
                raise UpstoxClientError("Malformed response from Upstox historical endpoint") from exc

        raise UpstoxClientError("Historical candle fetch failed unexpectedly")

    def _backoff(self, attempt: int) -> None:
        wait_seconds = self._backoff_seconds * (2 ** (attempt - 1))
        time.sleep(wait_seconds)

    @staticmethod
    def _parse_interval(interval: str) -> tuple[str, str]:
        if interval.endswith("minute"):
            count = interval.replace("minute", "")
            if count.isdigit():
                return ("minutes", count)
        if interval.endswith("hour"):
            count = interval.replace("hour", "")
            if count.isdigit():
                return ("hours", count)
        if interval == "day":
            return ("days", "1")
        if interval == "week":
            return ("weeks", "1")
        if interval == "month":
            return ("months", "1")
        raise UpstoxClientError(f"Unsupported interval format: {interval}")
