from __future__ import annotations

import logging

import pandas as pd

from tradeengine.core.data_processor import MarketDataProcessor


def _df_with_invalid_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:15:00+05:30",  # valid
                "2026-03-03T09:20:00+05:30",  # open > high
                "2026-03-03T09:25:00+05:30",  # negative volume
                "2026-03-03T09:30:00+05:30",  # close < low
            ],
            "open": [100.0, 110.0, 102.0, 103.0],
            "high": [101.0, 109.0, 103.0, 104.0],
            "low": [99.0, 108.0, 101.0, 103.5],
            "close": [100.5, 108.5, 102.5, 103.0],
            "volume": [1000.0, 1200.0, -5.0, 900.0],
        }
    )


def test_validate_logical_candles_drops_invalid_rows_and_logs(caplog) -> None:
    processor = MarketDataProcessor()

    with caplog.at_level(logging.WARNING):
        clean_df = processor.validate_logical_candles(_df_with_invalid_candles())

    assert len(clean_df) == 1
    assert any("[DATA_ERROR] Invalid OHLC" in msg for msg in caplog.messages)
    assert any("[DATA_WARNING] Dropped 3 invalid candles" in msg for msg in caplog.messages)


def test_validate_logical_candles_keeps_all_valid_rows() -> None:
    processor = MarketDataProcessor()
    df = pd.DataFrame(
        {
            "timestamp": ["2026-03-03T09:15:00+05:30", "2026-03-03T09:20:00+05:30"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1200.0],
        }
    )

    clean_df = processor.validate_logical_candles(df)
    assert len(clean_df) == 2
