from __future__ import annotations

import logging

import pandas as pd
import pytest

from tradeengine.core.data_processor import DataSchemaError, MarketDataProcessor


def _df_with_gap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:20:00+05:30",
                "2026-03-03T09:30:00+05:30",
                "2026-03-03T09:35:00+05:30",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )


def _df_no_gap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:20:00+05:30",
                "2026-03-03T09:25:00+05:30",
                "2026-03-03T09:30:00+05:30",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )


def test_validate_intervals_logs_missing_candles_warning(caplog) -> None:
    processor = MarketDataProcessor()

    with caplog.at_level(logging.WARNING):
        clean_df = processor.validate_intervals(_df_with_gap(), timeframe_minutes=5)

    assert len(clean_df) == 3
    assert any("[DATA_WARNING] Missing 1 candles" in msg for msg in caplog.messages)


def test_validate_intervals_no_warning_for_contiguous_data(caplog) -> None:
    processor = MarketDataProcessor()

    with caplog.at_level(logging.WARNING):
        clean_df = processor.validate_intervals(_df_no_gap(), timeframe_minutes=5)

    assert len(clean_df) == 3
    assert not any("[DATA_WARNING] Missing" in msg for msg in caplog.messages)


def test_validate_intervals_rejects_non_positive_timeframe() -> None:
    processor = MarketDataProcessor()

    with pytest.raises(DataSchemaError, match="timeframe_minutes must be positive"):
        processor.validate_intervals(_df_no_gap(), timeframe_minutes=0)
