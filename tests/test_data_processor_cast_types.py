from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.data_processor import DataSchemaError, MarketDataProcessor


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-03-03T09:15:00+05:30", "2026-03-03T09:20:00+05:30"],
            "open": ["100.1", "101.2"],
            "high": ["102.0", "103.0"],
            "low": ["99.8", "100.9"],
            "close": ["101.0", "102.3"],
            "volume": ["500", "600"],
        }
    )


def test_cast_types_converts_ohlcv_to_float64() -> None:
    processor = MarketDataProcessor()

    clean_df = processor.cast_types(_raw_df())

    assert str(clean_df["open"].dtype) == "float64"
    assert str(clean_df["high"].dtype) == "float64"
    assert str(clean_df["low"].dtype) == "float64"
    assert str(clean_df["close"].dtype) == "float64"
    assert str(clean_df["volume"].dtype) == "float64"


def test_cast_types_raises_on_nan_in_critical_price_columns() -> None:
    processor = MarketDataProcessor()
    df = _raw_df()
    df.loc[0, "open"] = "not-a-number"

    with pytest.raises(DataSchemaError, match="critical price columns"):
        processor.cast_types(df)


def test_cast_types_raises_on_nan_in_volume() -> None:
    processor = MarketDataProcessor()
    df = _raw_df()
    df.loc[1, "volume"] = "invalid"

    with pytest.raises(DataSchemaError, match="volume"):
        processor.cast_types(df)
