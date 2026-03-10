"""Centralized ML feature and target column configuration."""

ML_FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "label"
CLASS_LABELS = ["BUY", "SELL", "HOLD"]
