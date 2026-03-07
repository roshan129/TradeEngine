from __future__ import annotations

import math

import pandas as pd
import pytest

from tradeengine.core.features import FeatureEngineer, FeatureEngineeringError


def _trend_df(rows: int = 220) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]
    close = [100.0 + (0.5 * i) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1000.0 + i for i in range(rows)],
        }
    )


def _manual_ema_adjust_false(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = (alpha * value) + ((1.0 - alpha) * ema)
    return ema


def test_add_trend_features_creates_expected_columns_and_keeps_input_unchanged() -> None:
    engineer = FeatureEngineer()
    raw_df = _trend_df()

    out_df = engineer.add_trend_features(raw_df)

    for col in ["ema20", "ema50", "ema200", "vwap"]:
        assert col in out_df.columns

    assert raw_df.columns.tolist() == ["timestamp", "open", "high", "low", "close", "volume"]
    assert str(out_df["ema20"].dtype) == "float64"
    assert str(out_df["ema50"].dtype) == "float64"
    assert str(out_df["ema200"].dtype) == "float64"
    assert str(out_df["vwap"].dtype) == "float64"


def test_add_trend_features_ema_and_vwap_match_manual_calculation() -> None:
    engineer = FeatureEngineer()
    df = _trend_df()

    out_df = engineer.add_trend_features(df)

    idx = 19
    closes = df["close"].iloc[: idx + 1].tolist()
    manual_ema20 = _manual_ema_adjust_false(closes, span=20)
    assert math.isclose(out_df.loc[idx, "ema20"], manual_ema20, rel_tol=1e-9)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    manual_vwap_1 = (tp.iloc[:2] * df["volume"].iloc[:2]).sum() / df["volume"].iloc[:2].sum()
    assert math.isclose(out_df.loc[1, "vwap"], manual_vwap_1, rel_tol=1e-12)


def test_add_trend_features_rejects_non_numeric_ohlcv() -> None:
    engineer = FeatureEngineer()
    df = _trend_df(rows=10)
    df["close"] = df["close"].astype("object")
    df.loc[3, "close"] = "bad"

    with pytest.raises(FeatureEngineeringError, match="Non-numeric OHLCV"):
        engineer.add_trend_features(df)
