from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.data_processor import DataSchemaError, MarketDataProcessor


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:25:00+05:30",
                "2026-03-03T09:15:00+05:30",
                "2026-03-03T09:20:00+05:30",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )


def test_normalize_timestamp_converts_to_ist_and_sorts() -> None:
    processor = MarketDataProcessor()

    clean_df = processor.normalize_timestamp(_base_df())

    assert str(clean_df["timestamp"].dtype).startswith("datetime64[ns, Asia/Kolkata]")
    assert clean_df["timestamp"].is_monotonic_increasing


def test_normalize_timestamp_rejects_timezone_naive_values() -> None:
    processor = MarketDataProcessor()
    df = _base_df()
    df.loc[0, "timestamp"] = "2026-03-03 09:25:00"

    with pytest.raises(DataSchemaError, match="timezone-naive"):
        processor.normalize_timestamp(df)


def test_normalize_timestamp_rejects_duplicate_timestamps() -> None:
    processor = MarketDataProcessor()
    df = _base_df()
    df.loc[1, "timestamp"] = "2026-03-03T09:25:00+05:30"

    with pytest.raises(DataSchemaError, match="duplicate timestamps"):
        processor.normalize_timestamp(df)
