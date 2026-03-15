from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradeengine.ml.models.feature_config import CLASS_LABELS, ML_FEATURE_COLUMNS
from tradeengine.ml.models.predictor import ModelPredictionError, ModelPredictor
from tradeengine.ml.models.registry import ModelRegistry


class DummyModel:
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(x), 3), dtype=float)
        probs[:, 0] = 0.6
        probs[:, 1] = 0.2
        probs[:, 2] = 0.2
        return probs


def _feature_df(rows: int = 5) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "ema20": 100.0 + idx,
            "ema50": 99.5 + idx,
            "ema200": 99.0 + idx,
            "vwap": 100.1 + idx,
            "rsi": 45.0 + idx,
            "macd": np.sin(idx / 2.0),
            "macd_signal": np.cos(idx / 3.0),
            "macd_hist": np.sin(idx / 4.0),
            "roc": np.cos(idx / 5.0) * 0.01,
            "atr": 1.2 + idx * 0.01,
            "bb_width": 0.05 + idx * 0.001,
            "rolling_std": 0.6 + idx * 0.01,
            "dist_ema20": np.sin(idx / 6.0) * 0.01,
            "dist_vwap": np.cos(idx / 7.0) * 0.01,
            "higher_high": [True, False, True, False, True][:rows],
            "lower_low": [False, True, False, True, False][:rows],
            "rolling_volume_avg": 1000.0 + idx * 10.0,
            "minute_of_day": 555.0 + idx,
            "minutes_since_open": 0.0 + idx,
            "session_progress": np.clip(idx / max(rows - 1, 1), 0.0, 1.0),
            "gap_percent": 0.001 + idx * 0.0001,
            "distance_from_open": 0.002 + idx * 0.0001,
            "distance_from_previous_close": 0.0015 + idx * 0.0001,
        }
    )


def test_predictor_loads_and_predicts(tmp_path: Path) -> None:
    model_path = tmp_path / "model_v1.pkl"
    ModelRegistry().save(
        {
            "model": DummyModel(),
            "feature_columns": ML_FEATURE_COLUMNS,
            "class_labels": CLASS_LABELS,
        },
        str(model_path),
    )

    predictor = ModelPredictor(str(model_path))
    features = _feature_df(rows=5)

    predictions = predictor.predict(features)
    probabilities = predictor.predict_proba(features)

    assert len(predictions) == len(features)
    assert set(predictions.unique()) == {"BUY"}
    assert list(probabilities.columns) == [
        "buy_probability",
        "sell_probability",
        "hold_probability",
    ]
    assert probabilities.shape == (5, 3)


def test_predictor_rejects_missing_feature(tmp_path: Path) -> None:
    model_path = tmp_path / "model_v1.pkl"
    ModelRegistry().save(
        {
            "model": DummyModel(),
            "feature_columns": ML_FEATURE_COLUMNS,
            "class_labels": CLASS_LABELS,
        },
        str(model_path),
    )
    predictor = ModelPredictor(str(model_path))
    bad_features = _feature_df(rows=5).drop(columns=["ema20"])

    with pytest.raises(ModelPredictionError, match="Missing required feature columns"):
        predictor.predict(bad_features)
