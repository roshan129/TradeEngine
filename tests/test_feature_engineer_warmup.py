from __future__ import annotations

import logging

import pandas as pd
import pytest

from tradeengine.core.features import FeatureEngineer, FeatureEngineeringError


def _base_df(rows: int) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-03T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]
    close = [100.0 + (0.2 * i) for i in range(rows)]
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


def test_remove_initial_nan_rows_drops_nan_rows_and_logs(caplog) -> None:
    engineer = FeatureEngineer()
    df = _base_df(220)
    df["ema200"] = [float("nan")] * 199 + [1.0] * 21

    with caplog.at_level(logging.INFO):
        out_df = engineer.remove_initial_nan_rows(df)

    assert len(out_df) == 21
    assert not out_df.isna().any().any()
    assert any("[FEATURE_INFO] Removed 199 warmup rows" in msg for msg in caplog.messages)


def test_remove_initial_nan_rows_accepts_exactly_200_rows() -> None:
    engineer = FeatureEngineer()
    df = _base_df(200)
    df["ema200"] = [1.0] * 200

    out_df = engineer.remove_initial_nan_rows(df)
    assert len(out_df) == 200


def test_remove_initial_nan_rows_fails_gracefully_for_small_dataset() -> None:
    engineer = FeatureEngineer()
    df = _base_df(199)

    with pytest.raises(FeatureEngineeringError, match="Insufficient rows for feature warmup"):
        engineer.remove_initial_nan_rows(df)
