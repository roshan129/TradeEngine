from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, time
from zoneinfo import ZoneInfo

from tradeengine.market_data.models import Candle, normalize_candles
from tradeengine.market_data.upstox_client import UpstoxClient

logger = logging.getLogger(__name__)


class HistoricalDataService:
    def __init__(
        self,
        client: UpstoxClient,
        ist_timezone: str = "Asia/Kolkata",
        enforce_market_hours: bool = True,
    ) -> None:
        self._client = client
        self._ist_zone = ZoneInfo(ist_timezone)
        self._enforce_market_hours = enforce_market_hours

    def get_last_30_minutes_5min_candles(self, symbol: str) -> list[Candle]:
        now_ist = datetime.now(self._ist_zone)

        if self._enforce_market_hours and not self._is_market_open(now_ist):
            logger.warning("Market appears closed for IST timestamp=%s. Returning empty candles.", now_ist)
            return []

        from_ist = now_ist - timedelta(minutes=30)

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
