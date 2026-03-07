from __future__ import annotations

import numpy as np
import pandas as pd

from tradeengine.core.features import FeatureEngineer


def _volatility_df(rows: int = 260) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]

    close = []
    price = 100.0
    for i in range(rows):
        swing = np.sin(i / 8.0) * 1.2
        drift = 0.03 * i
        price = 100.0 + drift + swing
        close.append(float(price))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.2 for c in close],
            "high": [c + 0.8 for c in close],
            "low": [c - 0.9 for c in close],
            "close": close,
            "volume": [1300.0 + i for i in range(rows)],
        }
    )


def test_add_volatility_features_creates_expected_columns() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_volatility_features(_volatility_df())

    for col in ["atr", "bb_width", "rolling_std"]:
        assert col in out_df.columns


def test_atr_is_positive_after_warmup() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_volatility_features(_volatility_df())

    atr = out_df["atr"].dropna()
    assert not atr.empty
    assert (atr > 0).all()


def test_volatility_features_have_no_infinite_values() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_volatility_features(_volatility_df())

    subset = out_df[["atr", "bb_width", "rolling_std"]].dropna()
    assert not subset.empty
    assert np.isfinite(subset.to_numpy()).all()
