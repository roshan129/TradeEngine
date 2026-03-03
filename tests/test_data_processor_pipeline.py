from __future__ import annotations

import logging

import pandas as pd

from tradeengine.core.data_processor import MarketDataProcessor


def test_full_clean_pipeline_enforces_output_contract(caplog) -> None:
    processor = MarketDataProcessor()
    raw_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:25:00+05:30",
                "2026-03-03T09:15:00+05:30",
                "2026-03-03T09:25:00+05:30",  # duplicate timestamp
                "2026-03-03T09:35:00+05:30",  # gap from 09:25 -> 09:35
                "2026-03-03T09:40:00+05:30",
            ],
            "open": ["103", "100", "104", "106", "107"],
            "high": ["104", "101", "105", "107", "108"],
            "low": ["102", "99", "103", "105", "107"],
            "close": ["103.5", "100.5", "104.5", "106.5", "107.5"],
            "volume": ["1200", "1000", "1300", "1400", "1500"],
        }
    )

    with caplog.at_level(logging.INFO):
        clean_df = processor.full_clean_pipeline(raw_df)

    # duplicate should be removed; interval gap should be logged as warning.
    assert len(clean_df) == 4
    assert clean_df["timestamp"].is_monotonic_increasing
    assert clean_df["timestamp"].is_unique
    assert str(clean_df["timestamp"].dtype).startswith("datetime64[ns, Asia/Kolkata]")
    assert str(clean_df["open"].dtype) == "float64"
    assert str(clean_df["high"].dtype) == "float64"
    assert str(clean_df["low"].dtype) == "float64"
    assert str(clean_df["close"].dtype) == "float64"
    assert str(clean_df["volume"].dtype) == "float64"
    assert not clean_df[["open", "high", "low", "close", "volume"]].isna().any().any()
    assert any("[DATA_INFO] Removed 1 duplicate candles" in msg for msg in caplog.messages)
    assert any("[DATA_WARNING] Missing" in msg for msg in caplog.messages)


def test_full_clean_pipeline_drops_invalid_ohlc_rows() -> None:
    processor = MarketDataProcessor()
    raw_df = pd.DataFrame(
        {
            "timestamp": ["2026-03-03T09:15:00+05:30", "2026-03-03T09:20:00+05:30"],
            "open": [100.0, 110.0],
            "high": [101.0, 109.0],
            "low": [99.0, 108.0],
            "close": [100.5, 108.5],
            "volume": [1000.0, 1200.0],
        }
    )

    clean_df = processor.full_clean_pipeline(raw_df)
    assert len(clean_df) == 1
