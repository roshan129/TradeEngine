from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    """Serializable model evaluation report."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    by_class: dict[str, dict[str, float]]
    class_distribution: dict[str, int]
    feature_importance: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix,
            "by_class": self.by_class,
            "class_distribution": self.class_distribution,
            "feature_importance": self.feature_importance,
        }


def build_feature_importance(model: Any, feature_columns: list[str]) -> list[dict[str, float]]:
    raw_importance = getattr(model, "feature_importances_", None)
    if raw_importance is None:
        return []

    pairs = []
    for feature, importance in zip(feature_columns, raw_importance, strict=False):
        pairs.append({"feature": feature, "importance": float(importance)})
    pairs.sort(key=lambda item: item["importance"], reverse=True)
    return pairs


def _precision_recall_f1_for_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == label) & (y_pred == label)))
    fp = float(np.sum((y_true != label) & (y_pred == label)))
    fn = float(np.sum((y_true == label) & (y_pred != label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> list[list[int]]:
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    index = {label: idx for idx, label in enumerate(class_labels)}
    for truth, pred in zip(y_true, y_pred, strict=False):
        matrix[index[str(truth)], index[str(pred)]] += 1
    return matrix.tolist()


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    feature_importance: list[dict[str, float]] | None = None,
) -> ClassificationMetrics:
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    accuracy = float(np.mean(y_true == y_pred))

    by_class: dict[str, dict[str, float]] = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for label in class_labels:
        precision, recall, f1 = _precision_recall_f1_for_label(y_true, y_pred, label)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        by_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    distribution = {label: int((y_true == label).sum()) for label in class_labels}
    conf = _build_confusion_matrix(y_true, y_pred, class_labels)
    importance = feature_importance or []

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(np.mean(precision_values)),
        recall_macro=float(np.mean(recall_values)),
        f1_macro=float(np.mean(f1_values)),
        confusion_matrix=conf,
        by_class=by_class,
        class_distribution=distribution,
        feature_importance=importance,
    )


def format_metrics_summary(
    metrics: dict[str, Any],
    *,
    include_confusion_matrix: bool = True,
    feature_importance_top_n: int = 5,
) -> list[str]:
    lines = [
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision macro: {metrics['precision_macro']:.4f}",
        f"- Recall macro: {metrics['recall_macro']:.4f}",
        f"- F1 macro: {metrics['f1_macro']:.4f}",
    ]

    class_distribution = metrics.get("class_distribution", {})
    if class_distribution:
        lines.append(f"- Class distribution (test): {class_distribution}")

    by_class = metrics.get("by_class", {})
    for label in ("BUY", "SELL", "HOLD"):
        if label in by_class:
            scores = by_class[label]
            lines.append(
                f"- {label}: precision={scores['precision']:.4f}, "
                f"recall={scores['recall']:.4f}, f1={scores['f1']:.4f}"
            )

    if include_confusion_matrix:
        lines.append(f"- Confusion matrix: {metrics.get('confusion_matrix', [])}")

    feature_importance = metrics.get("feature_importance", [])
    if feature_importance:
        top = feature_importance[:feature_importance_top_n]
        lines.append(f"- Top {len(top)} features: {top}")

    return lines
