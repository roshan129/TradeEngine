from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.features import FeatureEngineer, FeatureEngineeringError


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T09:15:00+05:30",
                "2026-03-03T09:20:00+05:30",
                "2026-03-03T09:25:00+05:30",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


def test_prepare_base_dataframe_is_deterministic_and_non_mutating() -> None:
    engineer = FeatureEngineer()
    raw = _base_df()

    one = engineer.prepare_base_dataframe(raw)
    two = engineer.prepare_base_dataframe(raw)

    assert one.equals(two)
    assert one is not raw


def test_prepare_base_dataframe_rejects_unsorted_input() -> None:
    engineer = FeatureEngineer()
    raw = _base_df().iloc[[1, 0, 2]].reset_index(drop=True)

    with pytest.raises(FeatureEngineeringError, match="sorted ascending"):
        engineer.prepare_base_dataframe(raw)


def test_prepare_base_dataframe_rejects_duplicate_timestamps() -> None:
    engineer = FeatureEngineer()
    raw = _base_df()
    raw.loc[2, "timestamp"] = raw.loc[1, "timestamp"]

    with pytest.raises(FeatureEngineeringError, match="duplicate timestamps"):
        engineer.prepare_base_dataframe(raw)


def test_safe_shift_blocks_lookahead_shift() -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(FeatureEngineeringError, match="lookahead bias"):
        FeatureEngineer.safe_shift(series, periods=-1)
