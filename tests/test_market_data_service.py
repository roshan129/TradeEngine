from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
        start = datetime(2026, 2, 20, 3, 45, tzinfo=UTC)
        for i in range(600):
            ts = start + timedelta(minutes=5 * i)
            candles.append([ts.isoformat().replace("+00:00", "Z"), 10.0, 11.0, 9.0, 10.5, 100])
        return {"data": {"candles": candles}}


class MergeClient:
    def fetch_historical_candles(self, **_: object) -> dict[str, object]:
        return {
            "data": {
                "candles": [
                    ["2026-03-03T05:20:00Z", 10.0, 11.0, 9.0, 10.5, 100],
                    ["2026-03-03T05:25:00Z", 10.5, 11.5, 9.5, 11.0, 120],
                ]
            }
        }

    def fetch_intraday_candles(self, **_: object) -> dict[str, object]:
        return {
            "data": {
                "candles": [
                    ["2026-03-03T05:30:00Z", 11.0, 12.0, 10.8, 11.7, 150],
                ]
            }
        }


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


def test_anytime_method_uses_client_response_without_market_hours_gate() -> None:
    service = HistoricalDataService(client=ManyCandlesClient(), enforce_market_hours=True)
    candles = service.get_last_500_5min_candles_anytime(symbol="NSE_EQ|DUMMY")
    assert len(candles) == 500


def test_service_merges_historical_and_intraday_candles() -> None:
    service = HistoricalDataService(client=MergeClient(), enforce_market_hours=True)
    candles = service.get_last_500_5min_candles_anytime(symbol="NSE_EQ|DUMMY")

    assert len(candles) == 3
    assert candles[-1].timestamp.isoformat().startswith("2026-03-03T11:00:00+05:30")
