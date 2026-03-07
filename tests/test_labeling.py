from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradeengine.ml.dataset_builder import DatasetBuilder, DatasetBuildError
from tradeengine.ml.labeling import LabelGenerator


def test_generate_labels_future_return_and_assignment() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0, 100.0, 100.0]})
    out = LabelGenerator().generate_labels(
        df,
        horizon=1,
        buy_threshold=0.005,
        sell_threshold=-0.005,
    )

    assert len(out) == 3
    assert out["future_close"].tolist() == [101.0, 100.0, 100.0]
    assert out["label"].tolist() == ["BUY", "SELL", "HOLD"]


def test_generate_labels_horizon_shift_correctness() -> None:
    df = pd.DataFrame({"close": [10.0, 20.0, 30.0, 40.0]})
    out = LabelGenerator().generate_labels(df, horizon=2)

    assert len(out) == 2
    assert out.loc[0, "future_close"] == 30.0
    assert out.loc[1, "future_close"] == 40.0


def test_generate_multi_horizon_returns_columns() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    out = LabelGenerator().generate_multi_horizon_returns(df, horizons=(1, 2))

    assert "future_return_1" in out.columns
    assert "future_return_2" in out.columns
    assert out.loc[0, "future_return_1"] == pytest.approx(0.01)


def test_generate_volatility_adjusted_labels() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 99.0, 99.0],
            "atr": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = LabelGenerator().generate_volatility_adjusted_labels(df, horizon=1, atr_multiplier=0.5)

    assert out["label"].tolist() == ["BUY", "SELL", "HOLD"]


def _feature_df(rows: int = 60) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-01-01T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]

    close = np.array([100.0 + (0.2 * i) + (0.5 * np.sin(i / 5.0)) for i in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": np.array([1000.0 + i for i in range(rows)]),
            "ema20": close,
            "ema50": close - 0.1,
            "ema200": close - 0.2,
            "rsi": np.array([50.0 + ((i % 10) - 5) for i in range(rows)]),
            "macd": np.array([0.1 * np.sin(i / 6.0) for i in range(rows)]),
            "macd_signal": np.array([0.1 * np.sin(i / 7.0) for i in range(rows)]),
            "macd_hist": np.array([0.02 * np.cos(i / 5.0) for i in range(rows)]),
            "vwap": close - 0.05,
            "atr": np.array([1.2 for _ in range(rows)]),
            "bb_width": np.array([0.04 for _ in range(rows)]),
            "rolling_volume_avg": np.array([1005.0 + i for i in range(rows)]),
        }
    )


def test_dataset_builder_builds_valid_contract_dataset() -> None:
    out = DatasetBuilder().build_dataset(_feature_df(), horizons=(5, 10, 20), label_horizon=5)

    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].is_unique
    assert not out.isna().any().any()
    assert np.isfinite(out.select_dtypes(include=["number"]).to_numpy()).all()
    assert "future_close" not in out.columns
    assert "future_return" not in out.columns

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "vwap",
        "atr",
        "bb_width",
        "rolling_volume_avg",
        "future_return_5",
        "label",
    }
    assert required.issubset(set(out.columns))

    counts = DatasetBuilder.label_counts(out)
    assert counts["BUY"] + counts["SELL"] + counts["HOLD"] == len(out)


def test_dataset_builder_rejects_unsorted_timestamps() -> None:
    bad = _feature_df().iloc[::-1].reset_index(drop=True)

    with pytest.raises(DatasetBuildError, match="sorted ascending"):
        DatasetBuilder().build_dataset(bad)
