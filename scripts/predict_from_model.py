#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tradeengine.ml.models.predictor import ModelPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions using a saved ML model artifact")
    parser.add_argument("--model", required=True, help="Saved model artifact path")
    parser.add_argument("--input", required=True, help="Input features CSV path")
    parser.add_argument("--output", default="", help="Optional output CSV path")
    parser.add_argument(
        "--buy-threshold-proba",
        type=float,
        default=0.0,
        help="Minimum BUY probability for BUY signal (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--sell-threshold-proba",
        type=float,
        default=0.0,
        help="Minimum SELL probability for SELL signal (default: 0.0 = disabled)",
    )
    return parser.parse_args()


def _apply_threshold_gating(
    raw_predictions: pd.Series,
    probabilities: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.Series:
    if buy_threshold <= 0.0 and sell_threshold <= 0.0:
        return raw_predictions

    buy_proba = probabilities["buy_probability"]
    sell_proba = probabilities["sell_probability"]

    gated = pd.Series("HOLD", index=raw_predictions.index, name="prediction", dtype="object")
    buy_mask = buy_proba >= buy_threshold
    sell_mask = sell_proba >= sell_threshold

    gated.loc[buy_mask] = "BUY"
    gated.loc[sell_mask] = "SELL"

    conflict_mask = buy_mask & sell_mask
    if conflict_mask.any():
        gated.loc[conflict_mask] = "HOLD"
        gated.loc[conflict_mask & (buy_proba > sell_proba)] = "BUY"
        gated.loc[conflict_mask & (sell_proba > buy_proba)] = "SELL"
    return gated


def main() -> int:
    args = parse_args()
    for threshold_name, threshold_value in (
        ("buy-threshold-proba", args.buy_threshold_proba),
        ("sell-threshold-proba", args.sell_threshold_proba),
    ):
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(f"{threshold_name} must be in [0, 1], got {threshold_value}")

    input_df = pd.read_csv(args.input)

    predictor = ModelPredictor(model_path=args.model)
    raw_predictions = predictor.predict(input_df)
    probabilities = predictor.predict_proba(input_df)
    predictions = _apply_threshold_gating(
        raw_predictions=raw_predictions,
        probabilities=probabilities,
        buy_threshold=args.buy_threshold_proba,
        sell_threshold=args.sell_threshold_proba,
    )

    result = input_df.copy(deep=True)
    result["raw_prediction"] = raw_predictions
    result["prediction"] = predictions
    for column in probabilities.columns:
        result[column] = probabilities[column]

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Predictions written to: {args.output}")
    else:
        print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
