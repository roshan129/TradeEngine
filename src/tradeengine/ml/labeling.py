from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LabelingError(ValueError):
    """Raised when labeling input does not satisfy required contract."""


@dataclass(frozen=True)
class LabelGenerator:
    """Deterministic forward-label generator for ML datasets."""

    def _validate_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise LabelingError(f"horizon must be positive, received: {horizon}")

    @staticmethod
    def _validate_base_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
        if not isinstance(df, pd.DataFrame):
            raise LabelingError(f"Expected pandas DataFrame, got: {type(df).__name__}")
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise LabelingError(f"Missing required columns: {', '.join(missing)}")

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        buy_threshold: float = 0.003,
        sell_threshold: float = -0.003,
        label_column: str = "label",
        future_close_column: str = "future_close",
        future_return_column: str = "future_return",
    ) -> pd.DataFrame:
        """Generate fixed-threshold forward labels using close.shift(-horizon)."""
        self._validate_horizon(horizon)
        self._validate_base_columns(df, required=("close",))

        out = df.copy(deep=True)
        out[future_close_column] = pd.to_numeric(out["close"], errors="coerce").shift(-horizon)
        out[future_return_column] = (out[future_close_column] - out["close"]) / out["close"]

        out[label_column] = "HOLD"
        out.loc[out[future_return_column] > buy_threshold, label_column] = "BUY"
        out.loc[out[future_return_column] < sell_threshold, label_column] = "SELL"

        out = out.dropna(subset=[future_close_column, future_return_column]).reset_index(drop=True)
        return out

    def generate_multi_horizon_returns(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...] = (5, 10, 20),
    ) -> pd.DataFrame:
        """Add forward return columns for multiple future horizons."""
        self._validate_base_columns(df, required=("close",))
        if not horizons:
            raise LabelingError("horizons cannot be empty")

        out = df.copy(deep=True)
        for horizon in horizons:
            self._validate_horizon(horizon)
            future_close = out["close"].shift(-horizon)
            out[f"future_return_{horizon}"] = (future_close - out["close"]) / out["close"]

        return out

    def generate_volatility_adjusted_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        atr_multiplier: float = 0.5,
        label_column: str = "label",
        future_close_column: str = "future_close",
        future_move_column: str = "future_move",
    ) -> pd.DataFrame:
        """Generate labels using ATR-scaled future move thresholds."""
        self._validate_horizon(horizon)
        if atr_multiplier <= 0:
            raise LabelingError("atr_multiplier must be positive")
        self._validate_base_columns(df, required=("close", "atr"))

        out = df.copy(deep=True)
        out[future_close_column] = out["close"].shift(-horizon)
        out[future_move_column] = out[future_close_column] - out["close"]
        threshold = out["atr"] * atr_multiplier

        out[label_column] = "HOLD"
        out.loc[out[future_move_column] > threshold, label_column] = "BUY"
        out.loc[out[future_move_column] < -threshold, label_column] = "SELL"

        out = out.dropna(subset=[future_close_column, future_move_column]).reset_index(drop=True)
        return out
