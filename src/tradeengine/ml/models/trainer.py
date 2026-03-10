from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency import guard
    XGBClassifier = Any  # type: ignore[misc,assignment]
    _HAS_XGBOOST = False

from tradeengine.ml.models.evaluation import build_feature_importance, evaluate_classification
from tradeengine.ml.models.feature_config import CLASS_LABELS
from tradeengine.ml.models.registry import ModelRegistry


class ModelTrainingError(ValueError):
    """Raised when dataset/model training inputs are invalid."""


@dataclass(frozen=True)
class ModelTrainer:
    """Train deterministic multi-class XGBoost model from ML dataset."""

    feature_columns: list[str]
    target_column: str
    class_labels: list[str] | None = None
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        labels = self.class_labels if self.class_labels is not None else CLASS_LABELS
        object.__setattr__(self, "class_labels", list(labels))
        if not self.feature_columns:
            raise ModelTrainingError("feature_columns cannot be empty")
        if not 0.0 < self.test_fraction < 1.0:
            raise ModelTrainingError("test_fraction must be between 0 and 1")

    def load_dataset(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - direct passthrough for pandas IO errors
            raise ModelTrainingError(f"Failed to load dataset from {path}: {exc}") from exc

    def validate_dataset(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ModelTrainingError(f"Expected pandas DataFrame, got: {type(df).__name__}")

        required = [*self.feature_columns, self.target_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ModelTrainingError(f"Dataset missing required columns: {', '.join(missing)}")

        required_df = df[required]
        if required_df.isna().any().any():
            raise ModelTrainingError("Dataset has NaN values in features/target")

        bad_labels = sorted(set(df[self.target_column].unique()) - set(self.class_labels))
        if bad_labels:
            raise ModelTrainingError(f"Dataset has unsupported labels: {', '.join(bad_labels)}")

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if ts.isna().any():
                raise ModelTrainingError("Dataset contains invalid timestamp values")
            if not ts.is_monotonic_increasing:
                raise ModelTrainingError("Dataset timestamps must be sorted ascending")

    def split_dataset(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if len(df) < 2:
            raise ModelTrainingError("Dataset must contain at least 2 rows")

        split_index = int(len(df) * (1.0 - self.test_fraction))
        split_index = min(max(split_index, 1), len(df) - 1)

        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        x_train = self._prepare_features(train_df)
        y_train = train_df[self.target_column].astype(str)
        x_test = self._prepare_features(test_df)
        y_test = test_df[self.target_column].astype(str)
        return x_train, x_test, y_train, y_test

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df[self.feature_columns].copy(deep=True)
        for column in feature_df.columns:
            if pd.api.types.is_bool_dtype(feature_df[column]):
                feature_df[column] = feature_df[column].astype(int)
            feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        if feature_df.isna().any().any():
            raise ModelTrainingError("Feature matrix has non-numeric or NaN values")
        if not np.isfinite(feature_df.to_numpy()).all():
            raise ModelTrainingError("Feature matrix has non-finite values")
        return feature_df

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        if not _HAS_XGBOOST:
            raise ModelTrainingError("xgboost is required to train models. Install 'xgboost'.")

        label_to_int = {label: idx for idx, label in enumerate(self.class_labels)}
        try:
            y_encoded = y_train.map(label_to_int).astype(int)
        except Exception as exc:
            raise ModelTrainingError(f"Failed to encode target labels: {exc}") from exc
        if y_encoded.isna().any():
            raise ModelTrainingError("Target labels contain unexpected values")

        sample_weight = self.compute_balanced_sample_weight(y_encoded.to_numpy(dtype=int))

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.3,
            reg_alpha=0.2,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(x_train, y_encoded, sample_weight=sample_weight)
        return model

    def evaluate(
        self,
        model: XGBClassifier,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        raw_pred = model.predict(x_test)
        pred = np.array([self.class_labels[int(value)] for value in raw_pred], dtype=object)
        y_true = y_test.to_numpy(dtype=object)
        importance = build_feature_importance(model, self.feature_columns)

        metrics = evaluate_classification(
            y_true=y_true,
            y_pred=pred,
            class_labels=self.class_labels,
            feature_importance=importance,
        )
        return metrics.to_dict()

    def save_model(
        self,
        model: XGBClassifier,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bundle: dict[str, Any] = {
            "model": model,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "class_labels": self.class_labels,
        }
        return ModelRegistry().save(bundle=bundle, model_path=path, metadata=metadata)

    @staticmethod
    def compute_balanced_sample_weight(y_encoded: np.ndarray) -> np.ndarray:
        if y_encoded.size == 0:
            raise ModelTrainingError("Cannot compute sample weights for empty target array")

        values, counts = np.unique(y_encoded, return_counts=True)
        n_samples = int(y_encoded.size)
        n_classes = int(values.size)
        class_weight = {
            int(label): n_samples / (n_classes * int(count))
            for label, count in zip(values, counts, strict=False)
            if int(count) > 0
        }
        return np.array([class_weight[int(label)] for label in y_encoded], dtype=float)

    @staticmethod
    def walk_forward_indices(
        total_rows: int,
        initial_train_fraction: float = 0.6,
        test_fraction: float = 0.2,
        step_fraction: float = 0.1,
    ) -> list[tuple[int, int]]:
        if total_rows < 3:
            raise ModelTrainingError("walk-forward requires at least 3 rows")
        if not 0.0 < initial_train_fraction < 1.0:
            raise ModelTrainingError("initial_train_fraction must be between 0 and 1")
        if not 0.0 < test_fraction < 1.0:
            raise ModelTrainingError("test_fraction must be between 0 and 1")
        if not 0.0 < step_fraction < 1.0:
            raise ModelTrainingError("step_fraction must be between 0 and 1")

        initial_train_rows = max(1, int(total_rows * initial_train_fraction))
        test_rows = max(1, int(total_rows * test_fraction))
        step_rows = max(1, int(total_rows * step_fraction))

        folds: list[tuple[int, int]] = []
        train_end = initial_train_rows
        while train_end + test_rows <= total_rows:
            folds.append((train_end, train_end + test_rows))
            train_end += step_rows

        if not folds:
            raise ModelTrainingError("No walk-forward folds generated. Increase dataset size.")
        return folds
