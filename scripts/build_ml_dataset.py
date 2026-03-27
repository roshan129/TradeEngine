#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tradeengine.ml.dataset_builder import DatasetBuilder
from tradeengine.utils.paths import ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from engineered feature CSV")
    parser.add_argument("--input", required=True, help="Input feature CSV path")
    parser.add_argument(
        "--output",
        default="data/ml/ml_dataset.csv",
        help="Output ML CSV path (default: data/ml/ml_dataset.csv)",
    )
    parser.add_argument(
        "--horizons",
        default="5,10,20",
        help="Comma-separated future-return horizons (default: 5,10,20)",
    )
    parser.add_argument("--label-horizon", type=int, default=5, help="Horizon for label generation")
    parser.add_argument("--buy-threshold", type=float, default=0.003)
    parser.add_argument("--sell-threshold", type=float, default=-0.003)
    parser.add_argument(
        "--use-volatility-adjusted-labels",
        action="store_true",
        help="Use ATR-adjusted labels instead of fixed return thresholds",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=0.5,
        help="ATR multiplier for volatility-adjusted labeling (default: 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    raw_df = pd.read_csv(args.input)

    builder = DatasetBuilder()
    ml_df = builder.build_dataset(
        raw_df,
        horizons=horizons,
        label_horizon=args.label_horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        use_volatility_adjusted_labels=args.use_volatility_adjusted_labels,
        atr_multiplier=args.atr_multiplier,
    )

    counts = builder.label_counts(ml_df)
    ensure_parent_dir(args.output)
    ml_df.to_csv(args.output, index=False)

    print("ML Dataset Summary")
    print(f"- Input rows: {len(raw_df)}")
    print(f"- Output rows: {len(ml_df)}")
    print(f"- BUY: {counts['BUY']}")
    print(f"- SELL: {counts['SELL']}")
    print(f"- HOLD: {counts['HOLD']}")
    print(f"- Output CSV: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
