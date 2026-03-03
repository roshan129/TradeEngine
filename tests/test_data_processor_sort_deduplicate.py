from __future__ import annotations

import logging

import pandas as pd

from tradeengine.core.data_processor import MarketDataProcessor


def _raw_df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:20:00+05:30",
                "2026-03-03T09:15:00+05:30",
                "2026-03-03T09:20:00+05:30",
                "2026-03-03T09:25:00+05:30",
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )


def test_sort_and_deduplicate_orders_and_removes_duplicates(caplog) -> None:
    processor = MarketDataProcessor()

    with caplog.at_level(logging.INFO):
        clean_df = processor.sort_and_deduplicate(_raw_df_with_duplicates())

    assert clean_df["timestamp"].is_monotonic_increasing
    assert clean_df["timestamp"].is_unique
    assert len(clean_df) == 3
    assert any("[DATA_INFO] Removed 1 duplicate candles" in msg for msg in caplog.messages)


def test_sort_and_deduplicate_logs_nothing_when_no_duplicates(caplog) -> None:
    processor = MarketDataProcessor()
    df = _raw_df_with_duplicates().drop_duplicates(subset=["timestamp"], keep="first")

    with caplog.at_level(logging.INFO):
        clean_df = processor.sort_and_deduplicate(df)

    assert clean_df["timestamp"].is_unique
    assert not any("duplicate candles" in msg for msg in caplog.messages)
