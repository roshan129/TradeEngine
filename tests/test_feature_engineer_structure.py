from __future__ import annotations

import numpy as np
import pandas as pd

from tradeengine.core.features import FeatureEngineer


def _structure_df() -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    rows = 60
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]

    close = [100.0 + (0.2 * i) + (0.4 * np.sin(i / 4.0)) for i in range(rows)]
    high = [c + (1.0 if i % 2 == 0 else 0.6) for i, c in enumerate(close)]
    low = [c - (1.1 if i % 3 == 0 else 0.7) for i, c in enumerate(close)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.1 for c in close],
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0 + (i * 10.0) for i in range(rows)],
        }
    )


def test_add_structure_features_creates_expected_columns() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_structure_features(_structure_df())

    for col in [
        "dist_ema20",
        "dist_vwap",
        "higher_high",
        "lower_low",
        "rolling_volume_avg",
    ]:
        assert col in out_df.columns

    assert str(out_df["higher_high"].dtype) == "bool"
    assert str(out_df["lower_low"].dtype) == "bool"


def test_higher_high_and_lower_low_flags_match_previous_candle_logic() -> None:
    engineer = FeatureEngineer()
    df = _structure_df()
    out_df = engineer.add_structure_features(df)

    idx = 10
    assert out_df.loc[idx, "higher_high"] == (out_df.loc[idx, "high"] > out_df.loc[idx - 1, "high"])
    assert out_df.loc[idx, "lower_low"] == (out_df.loc[idx, "low"] < out_df.loc[idx - 1, "low"])


def test_structure_numeric_features_are_finite_after_warmup() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_structure_features(_structure_df())

    subset = out_df[["dist_ema20", "dist_vwap", "rolling_volume_avg"]].dropna()
    assert not subset.empty
    assert np.isfinite(subset.to_numpy()).all()
