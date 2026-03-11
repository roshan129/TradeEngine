"""ML dataset labeling and build utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from tradeengine.ml.dataset_builder import DatasetBuildError, DatasetBuilder
from tradeengine.ml.labeling import LabelGenerator, LabelingError

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ML_FEATURE_COLUMNS": ("tradeengine.ml.models.feature_config", "ML_FEATURE_COLUMNS"),
    "TARGET_COLUMN": ("tradeengine.ml.models.feature_config", "TARGET_COLUMN"),
    "CLASS_LABELS": ("tradeengine.ml.models.feature_config", "CLASS_LABELS"),
    "ModelTrainer": ("tradeengine.ml.models.trainer", "ModelTrainer"),
    "ModelTrainingError": ("tradeengine.ml.models.trainer", "ModelTrainingError"),
    "ModelPredictor": ("tradeengine.ml.models.predictor", "ModelPredictor"),
    "ModelPredictionError": ("tradeengine.ml.models.predictor", "ModelPredictionError"),
    "ModelRegistry": ("tradeengine.ml.models.registry", "ModelRegistry"),
    "ModelRegistryError": ("tradeengine.ml.models.registry", "ModelRegistryError"),
}

__all__ = [
    "LabelGenerator",
    "LabelingError",
    "DatasetBuilder",
    "DatasetBuildError",
    "ML_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "CLASS_LABELS",
    "ModelTrainer",
    "ModelTrainingError",
    "ModelPredictor",
    "ModelPredictionError",
    "ModelRegistry",
    "ModelRegistryError",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
