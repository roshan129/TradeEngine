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
    SESSION_CANDLE_COUNT = 75

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
        return self.get_last_trading_day_75_5min_candles_anytime(symbol=symbol)

    def get_last_trading_day_75_5min_candles_anytime(self, symbol: str) -> list[Candle]:
        now_ist = datetime.now(self._ist_zone)
        last_trading_day = self._resolve_last_trading_day(now_ist)

        # Fetch one full trading day window and keep exactly the latest 75 candles.
        session_start = datetime.combine(last_trading_day, time(hour=9, minute=15), tzinfo=self._ist_zone)
        session_end = datetime.combine(last_trading_day, time(hour=15, minute=30), tzinfo=self._ist_zone)

        raw = self._client.fetch_historical_candles(
            instrument_key=symbol,
            interval="5minute",
            from_datetime=session_start.astimezone(UTC),
            to_datetime=session_end.astimezone(UTC),
        )
        candles = normalize_candles(raw, timezone=str(self._ist_zone))
        return candles[-self.SESSION_CANDLE_COUNT :]

    def _fetch_last_500_5min_candles(self, symbol: str, enforce_market_hours: bool) -> list[Candle]:
        now_ist = datetime.now(self._ist_zone)

        if enforce_market_hours and not self._is_market_open(now_ist):
            logger.warning("Market appears closed for IST timestamp=%s. Returning empty candles.", now_ist)
            return []

        lookback_minutes = self.CANDLE_INTERVAL_MINUTES * self.TARGET_CANDLE_COUNT
        from_ist = now_ist - timedelta(minutes=lookback_minutes)
        raw = self._client.fetch_historical_candles(
            instrument_key=symbol,
            interval="5minute",
            from_datetime=from_ist.astimezone(UTC),
            to_datetime=now_ist.astimezone(UTC),
        )
        return normalize_candles(raw, timezone=str(self._ist_zone))

    @staticmethod
    def _is_market_open(current_ist: datetime) -> bool:
        if current_ist.weekday() >= 5:
            return False
        market_open = time(hour=9, minute=15)
        market_close = time(hour=15, minute=30)
        return market_open <= current_ist.time() <= market_close

    @staticmethod
    def _resolve_last_trading_day(current_ist: datetime):
        day = current_ist.date()
        if day.weekday() < 5 and current_ist.time() <= time(hour=15, minute=30):
            day -= timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day
