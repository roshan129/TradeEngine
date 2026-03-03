from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineeringError(ValueError):
    """Raised when feature engineering preconditions are not met."""


class FeatureEngineer:
    """Deterministic, leakage-safe feature engineering pipeline.

    Philosophy:
    - No lookahead bias: never use future candles for current-row features.
    - Deterministic transforms: same input -> same output.
    - No silent corrections: unsorted/duplicate timestamps fail fast.
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def prepare_base_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            msg = f"Expected pandas DataFrame, received: {type(df).__name__}"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            msg = f"Missing required columns for feature engineering: {', '.join(missing_columns)}"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        clean_df = df.copy(deep=True)
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        if clean_df["timestamp"].isna().any():
            msg = "Unparseable timestamps found in feature input"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if not clean_df["timestamp"].is_monotonic_increasing:
            msg = "Input candles must be sorted ascending by timestamp"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if clean_df["timestamp"].duplicated().any():
            msg = "Input candles contain duplicate timestamps"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        return clean_df

    @staticmethod
    def safe_shift(series: pd.Series, periods: int) -> pd.Series:
        if periods < 0:
            msg = "Negative shift is disallowed to prevent lookahead bias"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)
        return series.shift(periods)

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def remove_initial_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def full_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
