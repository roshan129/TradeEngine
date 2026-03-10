#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import MLSignalStrategy
from tradeengine.ml.dataset_builder import DatasetBuilder
from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.predictor import ModelPredictor
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-sample ML validation: train on first window, backtest last window."
    )
    parser.add_argument(
        "--features",
        default="feature_history_12m_5m.csv",
        help="Input feature history CSV (default: feature_history_12m_5m.csv)",
    )
    parser.add_argument(
        "--model-output",
        default="models/model_12m_5m_oos.pkl",
        help="Path to save OOS-trained model artifact",
    )
    parser.add_argument(
        "--output-dir",
        default="oos_results",
        help="Directory for OOS datasets, predictions, and backtest outputs",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Number of most-recent months reserved for out-of-sample testing (default: 3)",
    )
    parser.add_argument("--buy-threshold", type=float, default=0.002)
    parser.add_argument("--sell-threshold", type=float, default=-0.002)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
    )
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    return parser.parse_args()


def apply_threshold_gating(
    raw_predictions: pd.Series,
    probabilities: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.Series:
    buy_proba = probabilities["buy_probability"]
    sell_proba = probabilities["sell_probability"]

    gated = pd.Series("HOLD", index=raw_predictions.index, name="prediction", dtype="object")
    buy_mask = buy_proba >= buy_threshold
    sell_mask = sell_proba >= sell_threshold

    gated.loc[buy_mask] = "BUY"
    gated.loc[sell_mask] = "SELL"

    both = buy_mask & sell_mask
    if both.any():
        gated.loc[both] = "HOLD"
        gated.loc[both & (buy_proba > sell_proba)] = "BUY"
        gated.loc[both & (sell_proba > buy_proba)] = "SELL"
    return gated


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    if features["timestamp"].isna().any():
        raise ValueError("Input features contain invalid timestamps")
    features = features.sort_values("timestamp", ascending=True).reset_index(drop=True)

    end_ts = features["timestamp"].max()
    split_ts = end_ts - pd.DateOffset(months=args.test_months)
    train_features = features[features["timestamp"] <= split_ts].reset_index(drop=True)
    test_features = features[features["timestamp"] > split_ts].reset_index(drop=True)
    if train_features.empty or test_features.empty:
        raise ValueError("Train/test split is empty. Adjust test-months or input data range.")

    horizons = tuple(int(v.strip()) for v in args.horizons.split(",") if v.strip())
    builder = DatasetBuilder()
    train_ml = builder.build_dataset(
        train_features,
        horizons=horizons,
        label_horizon=args.label_horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    test_ml = builder.build_dataset(
        test_features,
        horizons=horizons,
        label_horizon=args.label_horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    train_ml_path = out_dir / "ml_train_oos.csv"
    test_ml_path = out_dir / "ml_test_oos.csv"
    train_ml.to_csv(train_ml_path, index=False)
    test_ml.to_csv(test_ml_path, index=False)

    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    trainer.validate_dataset(train_ml)
    trainer.validate_dataset(test_ml)
    x_train = trainer._prepare_features(train_ml)  # noqa: SLF001
    y_train = train_ml[TARGET_COLUMN].astype(str)
    x_test = trainer._prepare_features(test_ml)  # noqa: SLF001
    y_test = test_ml[TARGET_COLUMN].astype(str)

    model = trainer.train(x_train, y_train)
    holdout_metrics = trainer.evaluate(model, x_test, y_test)
    trainer.save_model(
        model,
        args.model_output,
        metadata={
            "model_version": "oos_9m_train_3m_test",
            "training_rows": int(len(train_ml)),
            "test_rows": int(len(test_ml)),
            "train_period_start": str(train_features["timestamp"].min()),
            "train_period_end": str(train_features["timestamp"].max()),
            "test_period_start": str(test_features["timestamp"].min()),
            "test_period_end": str(test_features["timestamp"].max()),
        },
    )

    predictor = ModelPredictor(args.model_output)
    thresholds = {
        "t2": (0.45, 0.55),
        "t3": (0.50, 0.60),
    }
    bt_config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_atr_multiple=args.stop_atr_multiple,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
        allow_shorts=True,
    )

    print("OOS Holdout Classification Summary")
    print(f"- Train rows (ML): {len(train_ml)}")
    print(f"- Test rows (ML): {len(test_ml)}")
    print(f"- Accuracy: {holdout_metrics['accuracy']:.4f}")
    print(f"- F1 macro: {holdout_metrics['f1_macro']:.4f}")
    print(
        f"- BUY recall: {holdout_metrics['by_class']['BUY']['recall']:.4f}, "
        f"SELL recall: {holdout_metrics['by_class']['SELL']['recall']:.4f}"
    )

    for tag, (buy_th, sell_th) in thresholds.items():
        raw_pred = predictor.predict(test_features)
        proba = predictor.predict_proba(test_features)
        gated = apply_threshold_gating(
            raw_pred,
            proba,
            buy_threshold=buy_th,
            sell_threshold=sell_th,
        )

        pred_df = test_features.copy(deep=True)
        pred_df["raw_prediction"] = raw_pred
        pred_df["prediction"] = gated
        pred_df["buy_probability"] = proba["buy_probability"]
        pred_df["sell_probability"] = proba["sell_probability"]
        pred_df["hold_probability"] = proba["hold_probability"]

        pred_path = out_dir / f"predictions_oos_{tag}.csv"
        pred_df.to_csv(pred_path, index=False)

        result = Backtester(
            strategy=MLSignalStrategy(allow_shorts=True),
            config=bt_config,
        ).run(pred_df)
        trades_path = out_dir / f"backtest_oos_{tag}_trades.csv"
        equity_path = out_dir / f"backtest_oos_{tag}_equity.csv"
        result.trades.to_csv(trades_path, index=False)
        result.equity_curve.to_csv(equity_path, index=False)

        signal_counts = pred_df["prediction"].value_counts().to_dict()
        print(f"OOS Backtest {tag.upper()} (buy>={buy_th:.2f}, sell>={sell_th:.2f})")
        print(f"- Signal counts: {signal_counts}")
        print(f"- Trades: {len(result.trades)}")
        print(f"- Total return %: {result.metrics['total_return_pct']:.4f}")
        print(f"- Max drawdown %: {result.metrics['max_drawdown_pct']:.4f}")
        print(f"- Win rate %: {result.metrics['win_rate']:.2f}")
        print(f"- Profit factor: {result.metrics['profit_factor']:.4f}")
        print(f"- Sharpe ratio: {result.metrics['sharpe_ratio']:.4f}")
        print(f"- Expectancy: {result.metrics['expectancy']:.4f}")
        print(f"- Trades CSV: {trades_path}")
        print(f"- Equity CSV: {equity_path}")
        print(f"- Predictions CSV: {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
