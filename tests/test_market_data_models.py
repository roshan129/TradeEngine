from __future__ import annotations

from tradeengine.market_data.models import normalize_candles


def test_normalize_candles_sorts_deduplicates_and_converts_types() -> None:
    payload = {
        "data": {
            "candles": [
                ["2026-03-02T05:25:00Z", "10.5", "11", "10", "10.9", "100"],
                ["2026-03-02T05:20:00Z", "9.5", "10", "9", "9.8", "90"],
                ["2026-03-02T05:20:00Z", "9.7", "10.1", "9.2", "9.9", "95"],
            ]
        }
    }

    candles = normalize_candles(payload)

    assert len(candles) == 2
    assert candles[0].timestamp.isoformat().endswith("+05:30")
    assert candles[0].open == 9.7
    assert candles[0].volume == 95
    assert candles[1].open == 10.5
