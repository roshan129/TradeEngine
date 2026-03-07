from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradeengine.core.features import FeatureEngineer, FeatureEngineeringError


def _pipeline_df(rows: int) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]

    close = [100.0 + (0.12 * i) + (0.9 * np.sin(i / 9.0)) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.2 for c in close],
            "high": [c + 0.8 for c in close],
            "low": [c - 0.9 for c in close],
            "close": close,
            "volume": [1000.0 + (i * 5.0) for i in range(rows)],
        }
    )


def test_full_feature_pipeline_enforces_final_contract() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.full_feature_pipeline(_pipeline_df(rows=500))

    expected_cols = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema20",
        "ema50",
        "ema200",
        "vwap",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "roc",
        "atr",
        "bb_width",
        "rolling_std",
        "dist_ema20",
        "dist_vwap",
        "higher_high",
        "lower_low",
        "rolling_volume_avg",
    }

    assert expected_cols.issubset(set(out_df.columns))
    assert out_df["timestamp"].is_monotonic_increasing
    assert out_df["timestamp"].is_unique
    assert not out_df.isna().any().any()
    assert np.isfinite(out_df.select_dtypes(include=["number"]).to_numpy()).all()
    assert str(out_df["higher_high"].dtype) == "bool"
    assert str(out_df["lower_low"].dtype) == "bool"


def test_full_feature_pipeline_fails_for_insufficient_dataset() -> None:
    engineer = FeatureEngineer()

    with pytest.raises(FeatureEngineeringError, match="Insufficient rows for feature warmup"):
        engineer.full_feature_pipeline(_pipeline_df(rows=199))
