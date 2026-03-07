from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradeengine.ml.labeling import LabelGenerator, LabelingError


class DatasetBuildError(ValueError):
    """Raised when ML dataset build contract checks fail."""


@dataclass(frozen=True)
class DatasetBuilder:
    """Build deterministic ML-ready datasets from engineered features."""

    label_generator: LabelGenerator = field(default_factory=LabelGenerator)

    REQUIRED_SCHEMA_COLUMNS: tuple[str, ...] = (
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
    )

    def _validate_sorted_unique_timestamp(self, df: pd.DataFrame) -> None:
        if "timestamp" not in df.columns:
            raise DatasetBuildError("Missing timestamp column")

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any():
            raise DatasetBuildError("Invalid timestamp values found")
        if not ts.is_monotonic_increasing:
            raise DatasetBuildError("Dataset timestamps must be sorted ascending")
        if ts.duplicated().any():
            raise DatasetBuildError("Dataset timestamps contain duplicates")

    @staticmethod
    def _validate_numeric_finite(df: pd.DataFrame) -> None:
        numeric = df.select_dtypes(include=["number"])
        if numeric.isna().any().any():
            raise DatasetBuildError("Dataset contains NaN values")
        if not np.isfinite(numeric.to_numpy()).all():
            raise DatasetBuildError("Dataset contains non-finite numeric values")

    def _validate_schema_contract(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_SCHEMA_COLUMNS if col not in df.columns]
        if missing:
            raise DatasetBuildError(
                f"Dataset missing required schema columns: {', '.join(missing)}"
            )

    def build_dataset(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...] = (5, 10, 20),
        label_horizon: int = 5,
        buy_threshold: float = 0.003,
        sell_threshold: float = -0.003,
        use_volatility_adjusted_labels: bool = False,
        atr_multiplier: float = 0.5,
    ) -> pd.DataFrame:
        """Build ML dataset: add multi-horizon returns, labels, validate, and remove leakage."""
        if not isinstance(df, pd.DataFrame):
            raise DatasetBuildError(f"Expected pandas DataFrame, got: {type(df).__name__}")
        if label_horizon not in horizons:
            raise DatasetBuildError("label_horizon must be included in horizons")

        out = df.copy(deep=True)
        self._validate_sorted_unique_timestamp(out)

        try:
            out = self.label_generator.generate_multi_horizon_returns(out, horizons=horizons)
            horizon_cols = [f"future_return_{h}" for h in horizons]
            out = out.dropna(subset=horizon_cols).reset_index(drop=True)
            label_return_column = f"future_return_{label_horizon}"

            if use_volatility_adjusted_labels:
                out = self.label_generator.generate_volatility_adjusted_labels(
                    out,
                    horizon=label_horizon,
                    atr_multiplier=atr_multiplier,
                    label_column="label",
                    future_close_column="future_close",
                    future_move_column="future_move",
                )
            else:
                out = self.label_generator.generate_labels(
                    out,
                    horizon=label_horizon,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    label_column="label",
                    future_close_column="future_close",
                    future_return_column=label_return_column,
                )
        except LabelingError as exc:
            raise DatasetBuildError(str(exc)) from exc

        leakage_columns = ["future_close", "future_return", "future_move"]
        existing_leakage = [col for col in leakage_columns if col in out.columns]
        if existing_leakage:
            out = out.drop(columns=existing_leakage)

        self._validate_sorted_unique_timestamp(out)
        self._validate_numeric_finite(out)
        self._validate_schema_contract(out)

        return out

    @staticmethod
    def label_counts(df: pd.DataFrame) -> dict[str, int]:
        """Return BUY/SELL/HOLD counts for class-imbalance checks."""
        if "label" not in df.columns:
            raise DatasetBuildError("Cannot compute label counts without 'label' column")

        counts = df["label"].value_counts().to_dict()
        return {
            "BUY": int(counts.get("BUY", 0)),
            "SELL": int(counts.get("SELL", 0)),
            "HOLD": int(counts.get("HOLD", 0)),
        }
