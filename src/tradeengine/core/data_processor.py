from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataSchemaError(ValueError):
    """Raised when market data does not satisfy the expected schema."""


class MarketDataProcessor:
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    # Canonical output dtype contract for the cleaned dataframe.
    CANONICAL_DTYPES: dict[str, str] = {
        "timestamp": "datetime64[ns]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    }
    IST_TIMEZONE = "Asia/Kolkata"

    def validate_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            msg = f"Expected pandas DataFrame, received: {type(df).__name__}"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        return df.copy(deep=True)

    def sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.validate_structure(df)
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")

        if clean_df["timestamp"].isna().any():
            msg = "Unable to sort/deduplicate because timestamp parsing failed"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        clean_df = clean_df.sort_values("timestamp", ascending=True).reset_index(drop=True)

        before_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        removed_count = before_count - len(clean_df)
        if removed_count > 0:
            logger.info("[DATA_INFO] Removed %s duplicate candles", removed_count)

        return clean_df

    def validate_intervals(self, df: pd.DataFrame, timeframe_minutes: int = 5) -> pd.DataFrame:
        if timeframe_minutes <= 0:
            msg = f"timeframe_minutes must be positive, got {timeframe_minutes}"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        clean_df = self.validate_structure(df)
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        if clean_df["timestamp"].isna().any():
            msg = "Unable to validate intervals because timestamp parsing failed"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        clean_df = clean_df.sort_values("timestamp", ascending=True).reset_index(drop=True)

        if len(clean_df) <= 1:
            return clean_df

        expected_delta = pd.Timedelta(minutes=timeframe_minutes)
        deltas = clean_df["timestamp"].diff()
        gap_rows = clean_df[deltas > expected_delta]

        for idx in gap_rows.index:
            current_ts = clean_df.loc[idx, "timestamp"]
            previous_ts = clean_df.loc[idx - 1, "timestamp"]
            missing_count = int((current_ts - previous_ts) / expected_delta) - 1
            if missing_count > 0:
                logger.warning(
                    "[DATA_WARNING] Missing %s candles between %s and %s",
                    missing_count,
                    previous_ts,
                    current_ts,
                )

        return clean_df

    def normalize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.validate_structure(df)

        naive_indices: list[int] = []
        for idx, value in clean_df["timestamp"].items():
            if value is None:
                continue
            if pd.isna(value):
                continue
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                naive_indices.append(int(idx))

        if naive_indices:
            msg = f"Found timezone-naive timestamps at rows: {naive_indices[:10]}"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        parsed = pd.to_datetime(clean_df["timestamp"], errors="coerce", utc=True)
        if parsed.isna().any():
            msg = "Found invalid timestamp values that cannot be parsed"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        clean_df["timestamp"] = parsed.dt.tz_convert(self.IST_TIMEZONE)
        clean_df = clean_df.sort_values("timestamp", ascending=True).reset_index(drop=True)

        duplicated_mask = clean_df["timestamp"].duplicated(keep=False)
        if duplicated_mask.any():
            duplicate_count = int(duplicated_mask.sum())
            msg = f"Found duplicate timestamps after normalization: {duplicate_count}"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        return clean_df

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.validate_structure(df)

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        critical_cols = ["open", "high", "low", "close"]
        bad_price_rows = clean_df[critical_cols].isna().any(axis=1)
        if bad_price_rows.any():
            bad_count = int(bad_price_rows.sum())
            logger.error("[DATA_ERROR] Found NaN in critical price columns for %s rows", bad_count)
            raise DataSchemaError("NaN values detected in critical price columns")

        volume_nan_rows = clean_df["volume"].isna()
        if volume_nan_rows.any():
            bad_count = int(volume_nan_rows.sum())
            logger.error("[DATA_ERROR] Found NaN in volume column for %s rows", bad_count)
            raise DataSchemaError("NaN values detected in volume column")

        clean_df[numeric_columns] = clean_df[numeric_columns].astype("float64")
        return clean_df

    def validate_logical_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.cast_types(df)
        timestamps = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        if timestamps.isna().any():
            msg = "Unable to validate candle logic because timestamp parsing failed"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        valid_mask = (
            (clean_df["high"] >= clean_df["low"])
            & (clean_df["open"] >= clean_df["low"])
            & (clean_df["open"] <= clean_df["high"])
            & (clean_df["close"] >= clean_df["low"])
            & (clean_df["close"] <= clean_df["high"])
            & (clean_df["volume"] >= 0)
        )

        invalid_df = clean_df[~valid_mask]
        if not invalid_df.empty:
            for idx in invalid_df.index:
                ts = timestamps.loc[idx]
                logger.error("[DATA_ERROR] Invalid OHLC at timestamp %s", ts)
            logger.warning("[DATA_WARNING] Dropped %s invalid candles", len(invalid_df))

        return clean_df[valid_mask].reset_index(drop=True)

    def full_clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = self.validate_structure(df)
        clean_df = self.sort_and_deduplicate(clean_df)
        clean_df = self.normalize_timestamp(clean_df)
        clean_df = self.cast_types(clean_df)
        clean_df = self.validate_logical_candles(clean_df)
        clean_df = self.validate_intervals(clean_df, timeframe_minutes=5)

        if clean_df["timestamp"].duplicated().any():
            msg = "Pipeline output contains duplicate timestamps"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        if not clean_df["timestamp"].is_monotonic_increasing:
            msg = "Pipeline output timestamps are not sorted ascending"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        if clean_df[["open", "high", "low", "close", "volume"]].isna().any().any():
            msg = "Pipeline output contains NaN values in OHLCV columns"
            logger.error("[DATA_ERROR] %s", msg)
            raise DataSchemaError(msg)

        for col, expected_dtype in self.CANONICAL_DTYPES.items():
            if col == "timestamp":
                if not isinstance(clean_df[col].dtype, pd.DatetimeTZDtype):
                    msg = "Pipeline output timestamp is not timezone-aware datetime"
                    logger.error("[DATA_ERROR] %s", msg)
                    raise DataSchemaError(msg)
                continue

            if str(clean_df[col].dtype) != expected_dtype:
                msg = f"Pipeline output column '{col}' has dtype {clean_df[col].dtype}, expected {expected_dtype}"
                logger.error("[DATA_ERROR] %s", msg)
                raise DataSchemaError(msg)

        return clean_df
