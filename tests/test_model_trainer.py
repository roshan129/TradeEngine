from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.trainer import ModelTrainer, ModelTrainingError


def _ml_df(rows: int = 180) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-03-01T09:15:00+05:30")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(rows)]
    idx = np.arange(rows, dtype=float)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ema20": 100.0 + (0.08 * idx),
            "ema50": 99.5 + (0.07 * idx),
            "ema200": 99.0 + (0.05 * idx),
            "vwap": 100.2 + (0.07 * idx),
            "rsi": 45.0 + (idx % 20),
            "macd": np.sin(idx / 8.0),
            "macd_signal": np.sin(idx / 9.0),
            "macd_hist": np.cos(idx / 10.0),
            "roc": np.sin(idx / 6.0) * 0.02,
            "atr": 1.5 + (idx % 5) * 0.01,
            "bb_width": 0.04 + (idx % 3) * 0.001,
            "rolling_std": 0.5 + (idx % 7) * 0.01,
            "dist_ema20": np.sin(idx / 11.0) * 0.01,
            "dist_vwap": np.cos(idx / 12.0) * 0.01,
            "higher_high": (idx % 2) == 0,
            "lower_low": (idx % 3) == 0,
            "rolling_volume_avg": 1200.0 + (idx * 3.0),
            "minute_of_day": 555.0 + idx,
            "minutes_since_open": idx,
            "session_progress": np.clip(idx / max(rows - 1, 1), 0.0, 1.0),
            "gap_percent": 0.001 + (idx * 0.0001),
            "distance_from_open": 0.002 + (idx * 0.0001),
            "distance_from_previous_close": 0.0015 + (idx * 0.0001),
        }
    )

    labels = np.where(idx % 3 == 0, "BUY", np.where(idx % 3 == 1, "SELL", "HOLD"))
    df[TARGET_COLUMN] = labels
    return df


def test_validate_dataset_rejects_missing_feature() -> None:
    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    bad_df = _ml_df().drop(columns=["ema20"])

    with pytest.raises(ModelTrainingError, match="(?i)missing required columns"):
        trainer.validate_dataset(bad_df)


def test_split_dataset_uses_time_order() -> None:
    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    df = _ml_df(rows=100)
    x_train, x_test, y_train, y_test = trainer.split_dataset(df)

    assert len(x_train) == 80
    assert len(x_test) == 20
    assert y_train.index.max() < y_test.index.min()


def test_training_evaluation_and_save(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")

    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    df = _ml_df(rows=180)
    trainer.validate_dataset(df)
    x_train, x_test, y_train, y_test = trainer.split_dataset(df)
    model = trainer.train(x_train, y_train)
    metrics = trainer.evaluate(model, x_test, y_test)

    assert "accuracy" in metrics
    assert "BUY" in metrics["by_class"]
    assert "feature_importance" in metrics

    model_path = tmp_path / "model_v1.pkl"
    metadata = trainer.save_model(
        model,
        str(model_path),
        metadata={"model_version": "v1", "training_rows": len(x_train)},
    )
    assert model_path.exists()
    assert (tmp_path / "model_v1_metadata.json").exists()
    assert metadata["model_version"] == "v1"


def test_compute_balanced_sample_weight_gives_higher_weight_to_minority() -> None:
    y_encoded = np.array([0, 0, 0, 1, 1, 2])
    weights = ModelTrainer.compute_balanced_sample_weight(y_encoded)

    # Class 2 appears least and should get highest weight.
    class_zero_weight = float(weights[y_encoded == 0][0])
    class_two_weight = float(weights[y_encoded == 2][0])
    assert class_two_weight > class_zero_weight


def test_walk_forward_indices_returns_time_ordered_folds() -> None:
    folds = ModelTrainer.walk_forward_indices(
        total_rows=100,
        initial_train_fraction=0.6,
        test_fraction=0.2,
        step_fraction=0.1,
    )

    assert folds[0] == (60, 80)
    assert folds[1] == (70, 90)
    assert folds[2] == (80, 100)
