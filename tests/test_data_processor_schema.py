from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.data_processor import DataSchemaError, MarketDataProcessor


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-03-03T09:15:00+05:30"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [5000.0],
        }
    )


def test_validate_structure_raises_when_required_column_missing() -> None:
    processor = MarketDataProcessor()
    df = _valid_df().drop(columns=["volume"])

    with pytest.raises(DataSchemaError, match="Missing required columns"):
        processor.validate_structure(df)


def test_validate_structure_raises_for_non_dataframe_input() -> None:
    processor = MarketDataProcessor()

    with pytest.raises(DataSchemaError, match="Expected pandas DataFrame"):
        processor.validate_structure([{"timestamp": "2026-03-03T09:15:00+05:30"}])  # type: ignore[arg-type]


def test_validate_structure_returns_deep_copy_when_valid() -> None:
    processor = MarketDataProcessor()
    raw_df = _valid_df()

    validated_df = processor.validate_structure(raw_df)

    assert list(validated_df.columns) == list(processor.REQUIRED_COLUMNS)
    assert validated_df is not raw_df
