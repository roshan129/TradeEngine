from __future__ import annotations

import math

import pandas as pd

from tradeengine.core.features import FeatureEngineer


def _momentum_df(rows: int = 260) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]

    close = []
    value = 100.0
    for i in range(rows):
        if i % 7 in (0, 1, 2):
            value += 0.8
        else:
            value -= 0.35
        close.append(value)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [c + 0.7 for c in close],
            "low": [c - 0.7 for c in close],
            "close": close,
            "volume": [1200.0 + i for i in range(rows)],
        }
    )


def test_add_momentum_features_creates_expected_columns() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_momentum_features(_momentum_df())

    for col in ["rsi", "macd", "macd_signal", "macd_hist", "roc"]:
        assert col in out_df.columns


def test_rsi_is_bounded_between_0_and_100_after_warmup() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_momentum_features(_momentum_df())

    valid_rsi = out_df["rsi"].dropna()
    assert not valid_rsi.empty
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_macd_histogram_is_macd_minus_signal() -> None:
    engineer = FeatureEngineer()
    out_df = engineer.add_momentum_features(_momentum_df())

    valid = out_df.dropna(subset=["macd", "macd_signal", "macd_hist"])
    assert not valid.empty

    sample = valid.iloc[-5:]
    for _, row in sample.iterrows():
        assert math.isclose(row["macd_hist"], row["macd"] - row["macd_signal"], rel_tol=1e-12)
