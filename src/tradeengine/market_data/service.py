from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, time
from zoneinfo import ZoneInfo

from tradeengine.market_data.models import Candle, normalize_candles
from tradeengine.market_data.upstox_client import UpstoxClient

logger = logging.getLogger(__name__)


class HistoricalDataService:
    CANDLE_INTERVAL_MINUTES = 5
    TARGET_CANDLE_COUNT = 500
    LOOKBACK_DAYS_FOR_500 = 20

    def __init__(
        self,
        client: UpstoxClient,
        ist_timezone: str = "Asia/Kolkata",
        enforce_market_hours: bool = True,
    ) -> None:
        self._client = client
        self._ist_zone = ZoneInfo(ist_timezone)
        self._enforce_market_hours = enforce_market_hours

    def get_last_500_5min_candles(self, symbol: str) -> list[Candle]:
        return self._fetch_last_500_5min_candles(
            symbol=symbol, enforce_market_hours=self._enforce_market_hours
        )

    def get_last_500_5min_candles_anytime(self, symbol: str) -> list[Candle]:
        return self._fetch_last_500_5min_candles(symbol=symbol, enforce_market_hours=False)

    def _fetch_last_500_5min_candles(self, symbol: str, enforce_market_hours: bool) -> list[Candle]:
        now_ist = datetime.now(self._ist_zone)

        if enforce_market_hours and not self._is_market_open(now_ist):
            logger.warning("Market appears closed for IST timestamp=%s. Returning empty candles.", now_ist)
            return []

        # Use a safe date window for 5-minute data, then trim to latest 500 candles.
        from_ist = now_ist - timedelta(days=self.LOOKBACK_DAYS_FOR_500)
        raw = self._client.fetch_historical_candles(
            instrument_key=symbol,
            interval="5minute",
            from_datetime=from_ist.astimezone(UTC),
            to_datetime=now_ist.astimezone(UTC),
        )
        intraday_raw: dict[str, object] = {}
        try:
            intraday_raw = self._client.fetch_intraday_candles(
                instrument_key=symbol,
                interval="5minute",
            )
        except Exception:
            logger.warning("Unable to fetch intraday candles; proceeding with historical candles only")

        merged_raw = self._merge_raw_payloads(raw, intraday_raw)
        candles = normalize_candles(merged_raw, timezone=str(self._ist_zone))
        return candles[-self.TARGET_CANDLE_COUNT :]

    @staticmethod
    def _is_market_open(current_ist: datetime) -> bool:
        if current_ist.weekday() >= 5:
            return False
        market_open = time(hour=9, minute=15)
        market_close = time(hour=15, minute=30)
        return market_open <= current_ist.time() <= market_close

    @staticmethod
    def _merge_raw_payloads(
        historical_payload: dict[str, object],
        intraday_payload: dict[str, object],
    ) -> dict[str, object]:
        historical_candles = historical_payload.get("data", {}).get("candles", [])
        intraday_candles = intraday_payload.get("data", {}).get("candles", [])

        merged_candles: list[object] = []
        if isinstance(historical_candles, list):
            merged_candles.extend(historical_candles)
        if isinstance(intraday_candles, list):
            merged_candles.extend(intraday_candles)

        return {"data": {"candles": merged_candles}}
