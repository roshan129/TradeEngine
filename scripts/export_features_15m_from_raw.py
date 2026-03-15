#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tradeengine.core.data_processor import MarketDataProcessor
from tradeengine.core.features import FeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample raw OHLCV CSV to 15-minute candles and export features."
    )
    parser.add_argument("--input", required=True, help="Raw OHLCV CSV path (5-minute or 1-minute)")
    parser.add_argument(
        "--output",
        default="feature_history_15m.csv",
        help="Output 15-minute features CSV path",
    )
    parser.add_argument(
        "--raw-output",
        default="",
        help="Optional output path for resampled 15-minute OHLCV CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    if raw_df["timestamp"].isna().any():
        raise ValueError("Input raw CSV has invalid timestamps")

    raw_df = raw_df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    raw_df = raw_df.set_index("timestamp")

    ohlc = raw_df[["open", "high", "low", "close"]].resample("15T").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    volume = raw_df[["volume"]].resample("15T").sum()
    resampled = pd.concat([ohlc, volume], axis=1).dropna().reset_index()

    processor = MarketDataProcessor()
    clean_df = processor.full_clean_pipeline(resampled, timeframe_minutes=15)
    features_df = FeatureEngineer().full_feature_pipeline(clean_df)
    features_df = features_df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    if args.raw_output:
        clean_df.to_csv(args.raw_output, index=False)

    features_df.to_csv(args.output, index=False)
    print(f"Saved 15-minute features CSV: {args.output} (rows={len(features_df)})")
    if args.raw_output:
        print(f"Saved 15-minute raw OHLCV CSV: {args.raw_output} (rows={len(clean_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
