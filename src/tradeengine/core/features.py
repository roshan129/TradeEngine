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
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_MULTIPLIER = 2.0
    VOLUME_ROLLING_PERIOD = 20
    SESSION_OPEN_MINUTE = (9 * 60) + 15
    SESSION_CLOSE_MINUTE = (15 * 60) + 30
    SESSION_DURATION_MINUTES = SESSION_CLOSE_MINUTE - SESSION_OPEN_MINUTE
    MIN_ROWS_FOR_FULL_FEATURES = 200
    FINAL_FEATURE_COLUMNS: tuple[str, ...] = (
        "ema20",
        "ema50",
        "ema200",
        "vwap",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "roc",
        "atr",
        "bb_width",
        "rolling_std",
        "dist_ema20",
        "dist_vwap",
        "higher_high",
        "lower_low",
        "rolling_volume_avg",
        "minute_of_day",
        "minutes_since_open",
        "session_progress",
        "gap_percent",
        "distance_from_open",
        "distance_from_previous_close",
    )

    def prepare_base_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate base candle schema/order and return a safe deep copy."""
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
        """Safe wrapper for pandas shift that blocks future-looking negative shifts."""
        if periods < 0:
            msg = "Negative shift is disallowed to prevent lookahead bias"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)
        return series.shift(periods)

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators: EMA20/50/200 and VWAP."""
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)

        close = clean_df["close"]
        clean_df["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        clean_df["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
        clean_df["ema200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()

        typical_price = (clean_df["high"] + clean_df["low"] + clean_df["close"]) / 3.0
        tpv = typical_price * clean_df["volume"]
        cumulative_volume = clean_df["volume"].cumsum()
        vwap = tpv.cumsum() / cumulative_volume
        zero_volume_mask = cumulative_volume == 0
        if zero_volume_mask.any():
            # Index data can report zero volume; fall back to typical price for VWAP.
            vwap = vwap.copy()
            vwap.loc[zero_volume_mask] = typical_price.loc[zero_volume_mask]
        clean_df["vwap"] = vwap
        return clean_df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators: RSI, MACD (+signal/hist), and ROC."""
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
        """Add volatility indicators: ATR, Bollinger width, and rolling std."""
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)

        prev_close = self.safe_shift(clean_df["close"], periods=1)
        tr_components = pd.concat(
            [
                clean_df["high"] - clean_df["low"],
                (clean_df["high"] - prev_close).abs(),
                (clean_df["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        clean_df["atr"] = true_range.ewm(
            alpha=1 / self.ATR_PERIOD,
            min_periods=self.ATR_PERIOD,
            adjust=False,
        ).mean()

        rolling_mean = clean_df["close"].rolling(
            window=self.BB_PERIOD,
            min_periods=self.BB_PERIOD,
        ).mean()
        rolling_std = clean_df["close"].rolling(
            window=self.BB_PERIOD,
            min_periods=self.BB_PERIOD,
        ).std(ddof=0)
        upper_band = rolling_mean + (self.BB_STD_MULTIPLIER * rolling_std)
        lower_band = rolling_mean - (self.BB_STD_MULTIPLIER * rolling_std)

        bb_width = ((upper_band - lower_band) / rolling_mean).astype("float64")
        bb_width = bb_width.replace([float("inf"), float("-inf")], pd.NA)
        clean_df["bb_width"] = bb_width

        clean_df["rolling_std"] = rolling_std.astype("float64")

        finite_cols = ["atr", "bb_width", "rolling_std"]
        inf_mask = clean_df[finite_cols].isin([float("inf"), float("-inf")]).any(axis=1)
        if inf_mask.any():
            msg = "Infinite values detected in volatility features"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        return clean_df

    def add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual structure features and price-distance derived metrics."""
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)

        if "ema20" not in clean_df.columns or "vwap" not in clean_df.columns:
            clean_df = self.add_trend_features(clean_df)

        clean_df["dist_ema20"] = clean_df["close"] - clean_df["ema20"]
        clean_df["dist_vwap"] = clean_df["close"] - clean_df["vwap"]

        prev_high = self.safe_shift(clean_df["high"], periods=1)
        prev_low = self.safe_shift(clean_df["low"], periods=1)
        clean_df["higher_high"] = (clean_df["high"] > prev_high).fillna(False)
        clean_df["lower_low"] = (clean_df["low"] < prev_low).fillna(False)

        clean_df["rolling_volume_avg"] = clean_df["volume"].rolling(
            window=self.VOLUME_ROLLING_PERIOD,
            min_periods=self.VOLUME_ROLLING_PERIOD,
        ).mean()

        inf_mask = clean_df[["dist_ema20", "dist_vwap", "rolling_volume_avg"]].isin(
            [float("inf"), float("-inf")]
        )
        if inf_mask.any().any():
            msg = "Infinite values detected in structure features"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        clean_df["higher_high"] = clean_df["higher_high"].astype("bool")
        clean_df["lower_low"] = clean_df["lower_low"].astype("bool")
        return clean_df

    def add_time_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday time context + opening gap behavior features."""
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self._coerce_numeric_ohlcv(clean_df)
        timestamps = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise FeatureEngineeringError(
                "Unparseable timestamps found while building time features"
            )

        minute_of_day = (timestamps.dt.hour * 60) + timestamps.dt.minute
        clean_df["minute_of_day"] = minute_of_day.astype("float64")
        clean_df["minutes_since_open"] = (
            (minute_of_day - self.SESSION_OPEN_MINUTE).clip(lower=0)
        ).astype("float64")
        session_progress = clean_df["minutes_since_open"] / float(self.SESSION_DURATION_MINUTES)
        clean_df["session_progress"] = session_progress.clip(lower=0.0, upper=1.0).astype("float64")

        session_date = timestamps.dt.date
        day_open = clean_df.groupby(session_date, sort=False)["open"].transform("first")
        day_close = clean_df.groupby(session_date, sort=False)["close"].last()
        previous_day_close = session_date.map(day_close.shift(1))

        clean_df["distance_from_open"] = (clean_df["close"] - day_open).astype("float64")
        clean_df["distance_from_previous_close"] = (
            clean_df["close"] - previous_day_close
        ).astype("float64")
        gap_percent = ((day_open - previous_day_close) / previous_day_close) * 100.0
        clean_df["gap_percent"] = gap_percent.astype("float64")

        time_cols = [
            "gap_percent",
            "distance_from_open",
            "distance_from_previous_close",
        ]
        clean_df[time_cols] = clean_df[time_cols].fillna(0.0)

        inf_mask = clean_df[
            [
                "minute_of_day",
                "minutes_since_open",
                "session_progress",
                "gap_percent",
                "distance_from_open",
                "distance_from_previous_close",
            ]
        ].isin([float("inf"), float("-inf")])
        if inf_mask.any().any():
            raise FeatureEngineeringError("Infinite values detected in time/context features")

        return clean_df

    def remove_initial_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop warmup rows containing NaN after indicator computation."""
        clean_df = self.prepare_base_dataframe(df)

        if len(clean_df) < self.MIN_ROWS_FOR_FULL_FEATURES:
            msg = (
                f"Insufficient rows for feature warmup: received {len(clean_df)}, "
                f"need at least {self.MIN_ROWS_FOR_FULL_FEATURES}"
            )
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        nan_rows = clean_df.isna().any(axis=1)
        removed_count = int(nan_rows.sum())
        if removed_count > 0:
            logger.info("[FEATURE_INFO] Removed %s warmup rows containing NaN", removed_count)

        clean_df = clean_df.dropna().reset_index(drop=True)
        return clean_df

    def full_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature stages and enforce final no-NaN/no-dup contract."""
        clean_df = self.prepare_base_dataframe(df)
        clean_df = self.add_trend_features(clean_df)
        clean_df = self.add_momentum_features(clean_df)
        clean_df = self.add_volatility_features(clean_df)
        clean_df = self.add_structure_features(clean_df)
        clean_df = self.add_time_context_features(clean_df)
        clean_df = self.remove_initial_nan_rows(clean_df)

        missing_features = [
            col for col in self.FINAL_FEATURE_COLUMNS if col not in clean_df.columns
        ]
        if missing_features:
            msg = f"Missing required engineered features: {', '.join(missing_features)}"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if clean_df["timestamp"].duplicated().any():
            msg = "Feature pipeline output contains duplicate timestamps"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if not clean_df["timestamp"].is_monotonic_increasing:
            msg = "Feature pipeline output timestamps are not sorted ascending"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if clean_df.isna().any().any():
            msg = "Feature pipeline output contains NaN values"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        numeric_cols = [*self.NUMERIC_COLUMNS, *self.FINAL_FEATURE_COLUMNS]
        numeric_cols = [col for col in numeric_cols if col not in {"higher_high", "lower_low"}]
        inf_mask = clean_df[numeric_cols].isin([float("inf"), float("-inf")]).any(axis=1)
        if inf_mask.any():
            msg = "Feature pipeline output contains infinite values"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        if (
            str(clean_df["higher_high"].dtype) != "bool"
            or str(clean_df["lower_low"].dtype) != "bool"
        ):
            msg = "Feature flag columns must be bool dtype"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        logger.info("[FEATURE_INFO] Full feature pipeline produced %s rows", len(clean_df))
        return clean_df

    def _coerce_numeric_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert base OHLCV columns to float64 and reject non-numeric data."""
        clean_df = df.copy(deep=True)
        for col in self.NUMERIC_COLUMNS:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        if clean_df[list(self.NUMERIC_COLUMNS)].isna().any().any():
            msg = "Non-numeric OHLCV values found in feature input"
            logger.error("[FEATURE_ERROR] %s", msg)
            raise FeatureEngineeringError(msg)

        clean_df[list(self.NUMERIC_COLUMNS)] = clean_df[list(self.NUMERIC_COLUMNS)].astype(
            "float64"
        )
        return clean_df
