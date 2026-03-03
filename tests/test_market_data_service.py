from __future__ import annotations

from datetime import datetime

from tradeengine.market_data.service import HistoricalDataService


class DummyClient:
    def __init__(self) -> None:
        self.called = False

    def fetch_historical_candles(self, **_: object) -> dict[str, object]:
        self.called = True
        return {
            "data": {
                "candles": [
                    ["2026-03-03T05:25:00Z", 10.0, 11.0, 9.5, 10.5, 100],
                ]
            }
        }


class ManyCandlesClient:
    def fetch_historical_candles(self, **_: object) -> dict[str, object]:
        candles = []
        for i in range(80):
            minute = 15 + (i * 5)
            hour = 9 + (minute // 60)
            minute = minute % 60
            candles.append([f"2026-03-02T{hour:02d}:{minute:02d}:00Z", 10.0, 11.0, 9.0, 10.5, 100])
        return {"data": {"candles": candles}}


def test_anytime_method_bypasses_market_hours(monkeypatch) -> None:
    client = DummyClient()
    service = HistoricalDataService(client=client, enforce_market_hours=True)

    monkeypatch.setattr(
        HistoricalDataService,
        "_is_market_open",
        staticmethod(lambda _: False),
    )

    blocked = service.get_last_500_5min_candles(symbol="NSE_EQ|DUMMY")
    assert blocked == []
    assert client.called is False

    open_anytime = service.get_last_500_5min_candles_anytime(symbol="NSE_EQ|DUMMY")
    assert len(open_anytime) == 1
    assert isinstance(open_anytime[0].timestamp, datetime)
    assert client.called is True


def test_anytime_method_returns_latest_75_candles() -> None:
    service = HistoricalDataService(client=ManyCandlesClient(), enforce_market_hours=True)
    candles = service.get_last_trading_day_75_5min_candles_anytime(symbol="NSE_EQ|DUMMY")
    assert len(candles) == 75
