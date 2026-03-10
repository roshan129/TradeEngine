from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tradeengine.ml.models.feature_config import CLASS_LABELS, ML_FEATURE_COLUMNS
from tradeengine.ml.models.registry import ModelRegistry


class ModelPredictionError(ValueError):
    """Raised when prediction requests violate model input contract."""


@dataclass(frozen=True)
class ModelPredictor:
    """Load and run inference against saved model artifacts."""

    model_path: str

    def __post_init__(self) -> None:
        bundle = ModelRegistry.load_bundle(self.model_path)
        object.__setattr__(self, "_model", bundle["model"])
        object.__setattr__(
            self,
            "_feature_columns",
            bundle.get("feature_columns", ML_FEATURE_COLUMNS),
        )
        object.__setattr__(self, "_class_labels", bundle.get("class_labels", CLASS_LABELS))

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def class_labels(self) -> list[str]:
        return list(self._class_labels)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise ModelPredictionError(f"Expected pandas DataFrame, got: {type(features).__name__}")

        missing = [col for col in self._feature_columns if col not in features.columns]
        if missing:
            raise ModelPredictionError(f"Missing required feature columns: {', '.join(missing)}")

        out = features[self._feature_columns].copy(deep=True)
        for column in out.columns:
            if pd.api.types.is_bool_dtype(out[column]):
                out[column] = out[column].astype(int)
            out[column] = pd.to_numeric(out[column], errors="coerce")

        if out.isna().any().any():
            raise ModelPredictionError("Prediction features contain NaN or non-numeric values")
        if not np.isfinite(out.to_numpy()).all():
            raise ModelPredictionError("Prediction features contain non-finite values")
        return out

    def predict(self, features: pd.DataFrame) -> pd.Series:
        prepared = self._prepare_features(features)
        raw = self._model.predict(prepared)
        labels = [self._decode_class(pred) for pred in raw]
        return pd.Series(labels, index=prepared.index, name="prediction", dtype="object")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_features(features)
        probabilities = self._model.predict_proba(prepared)
        if probabilities.shape[1] != len(self._class_labels):
            raise ModelPredictionError("Model class count does not match configured class labels")

        columns = [f"{label.lower()}_probability" for label in self._class_labels]
        return pd.DataFrame(probabilities, index=prepared.index, columns=columns)

    def _decode_class(self, encoded_prediction: Any) -> str:
        if isinstance(encoded_prediction, str):
            return encoded_prediction
        idx = int(encoded_prediction)
        if idx < 0 or idx >= len(self._class_labels):
            raise ModelPredictionError(f"Predicted class index out of range: {idx}")
        return self._class_labels[idx]
