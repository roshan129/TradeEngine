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
    NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    RSI_PERIOD = 14
    ROC_PERIOD = 12
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

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
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)

        close = clean_df["close"]
        clean_df["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        clean_df["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
        clean_df["ema200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()

        typical_price = (clean_df["high"] + clean_df["low"] + clean_df["close"]) / 3.0
        tpv = typical_price * clean_df["volume"]
        cumulative_volume = clean_df["volume"].cumsum()
        clean_df["vwap"] = tpv.cumsum() / cumulative_volume
        return clean_df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)

        close = clean_df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(
            alpha=1 / self.RSI_PERIOD, min_periods=self.RSI_PERIOD, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / self.RSI_PERIOD, min_periods=self.RSI_PERIOD, adjust=False
        ).mean()

        rs = avg_gain / avg_loss
        clean_df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        clean_df.loc[avg_loss == 0, "rsi"] = 100.0
        clean_df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50.0

        ema_fast = close.ewm(span=self.MACD_FAST, adjust=False, min_periods=self.MACD_FAST).mean()
        ema_slow = close.ewm(span=self.MACD_SLOW, adjust=False, min_periods=self.MACD_SLOW).mean()
        clean_df["macd"] = ema_fast - ema_slow
        clean_df["macd_signal"] = clean_df["macd"].ewm(
            span=self.MACD_SIGNAL, adjust=False, min_periods=self.MACD_SIGNAL
        ).mean()
        clean_df["macd_hist"] = clean_df["macd"] - clean_df["macd_signal"]

        clean_df["roc"] = close.pct_change(periods=self.ROC_PERIOD) * 100.0
        return clean_df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def remove_initial_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def full_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _coerce_numeric_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy(deep=True)
        for col in self.NUMERIC_COLUMNS:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        if clean_df[list(self.NUMERIC_COLUMNS)].isna().any().any():
            msg = "Non-numeric OHLCV values found in feature input"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        clean_df[list(self.NUMERIC_COLUMNS)] = clean_df[list(self.NUMERIC_COLUMNS)].astype("float64")
        return clean_df
