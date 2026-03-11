#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time
from itertools import product
from pathlib import Path
from time import strptime
from typing import Any

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import MLSignalStrategy
from tradeengine.ml.dataset_builder import DatasetBuilder
from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.predictor import ModelPredictor
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ML thresholds/risk/stop on train window, then evaluate on OOS window."
    )
    parser.add_argument("--features", default="feature_history_12m_5m_v2.csv")
    parser.add_argument("--output-dir", default="oos_tune_results")
    parser.add_argument("--model-output", default="models/model_12m_5m_tuned_oos.pkl")
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--buy-threshold", type=float, default=0.002)
    parser.add_argument("--sell-threshold", type=float, default=-0.002)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument(
        "--buy-threshold-grid",
        default="0.40,0.45,0.50",
        help="Comma-separated probability thresholds for BUY gating",
    )
    parser.add_argument(
        "--sell-threshold-grid",
        default="0.55,0.60,0.65",
        help="Comma-separated probability thresholds for SELL gating",
    )
    parser.add_argument(
        "--risk-grid",
        default="0.005,0.01,0.015",
        help="Comma-separated risk_per_trade values",
    )
    parser.add_argument(
        "--stop-grid",
        default="0.8,1.0,1.2",
        help="Comma-separated stop_atr_multiple values",
    )
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    parser.add_argument("--ml-entry-start", default="09:20")
    parser.add_argument("--ml-entry-end", default="10:20")
    parser.add_argument("--train-session-start", default="09:20")
    parser.add_argument("--train-session-end", default="10:20")
    return parser.parse_args()


def parse_float_grid(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_hhmm(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.split(":", 1)
    return int(hour_text), int(minute_text)


def _parse_hhmm_time(value: str) -> time:
    hour_text, minute_text = value.split(":", 1)
    return time(hour=int(hour_text), minute=int(minute_text))


def _filter_by_session(
    df: pd.DataFrame,
    start: time,
    end: time,
    *,
    label: str,
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        print(f"[WARN] No timestamp column; skipping {label} session filter.")
        return df

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label} data has invalid timestamps; cannot apply session filter.")

    times = timestamps.dt.time
    if start <= end:
        mask = (times >= start) & (times <= end)
    else:
        mask = (times >= start) | (times <= end)

    filtered = df.loc[mask].copy()
    print(
        f"[INFO] {label} session filter {start.strftime('%H:%M')}-{end.strftime('%H:%M')}: "
        f"{len(df)} -> {len(filtered)} rows"
    )
    return filtered


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


def build_prediction_df(
    predictor: ModelPredictor,
    base_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    raw_pred = predictor.predict(base_df)
    proba = predictor.predict_proba(base_df)
    gated = apply_threshold_gating(raw_pred, proba, buy_threshold, sell_threshold)

    out = base_df.copy(deep=True)
    out["raw_prediction"] = raw_pred
    out["prediction"] = gated
    out["buy_probability"] = proba["buy_probability"]
    out["sell_probability"] = proba["sell_probability"]
    out["hold_probability"] = proba["hold_probability"]
    return out


def score_metrics(metrics: dict[str, float]) -> float:
    return float(metrics["total_return_pct"] - (0.5 * metrics["max_drawdown_pct"]))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    if features["timestamp"].isna().any():
        raise ValueError("Invalid timestamps in features CSV")
    features = features.sort_values("timestamp", ascending=True).reset_index(drop=True)

    end_ts = features["timestamp"].max()
    split_ts = end_ts - pd.DateOffset(months=args.test_months)
    train_features = features[features["timestamp"] <= split_ts].reset_index(drop=True)
    test_features = features[features["timestamp"] > split_ts].reset_index(drop=True)
    if train_features.empty or test_features.empty:
        raise ValueError("Train/test split is empty. Adjust test-months.")

    session_start = _parse_hhmm_time(args.train_session_start)
    session_end = _parse_hhmm_time(args.train_session_end)
    train_features = _filter_by_session(
        train_features,
        session_start,
        session_end,
        label="Training features",
    )

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

    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    x_train = trainer._prepare_features(train_ml)  # noqa: SLF001
    y_train = train_ml[TARGET_COLUMN].astype(str)
    x_test = trainer._prepare_features(test_ml)  # noqa: SLF001
    y_test = test_ml[TARGET_COLUMN].astype(str)

    model = trainer.train(x_train, y_train)
    trainer.save_model(
        model,
        args.model_output,
        metadata={
            "model_version": "oos_tuned",
            "training_rows": int(len(train_ml)),
            "test_rows": int(len(test_ml)),
        },
    )
    holdout_metrics = trainer.evaluate(model, x_test, y_test)

    predictor = ModelPredictor(args.model_output)
    buy_grid = parse_float_grid(args.buy_threshold_grid)
    sell_grid = parse_float_grid(args.sell_threshold_grid)
    risk_grid = parse_float_grid(args.risk_grid)
    stop_grid = parse_float_grid(args.stop_grid)

    start_h, start_m = parse_hhmm(args.ml_entry_start)
    end_h, end_m = parse_hhmm(args.ml_entry_end)
    entry_start = strptime(f"{start_h:02d}:{start_m:02d}", "%H:%M")
    entry_end = strptime(f"{end_h:02d}:{end_m:02d}", "%H:%M")
    strategy = MLSignalStrategy(
        allow_shorts=True,
        entry_session_start=pd.Timestamp(
            year=2000,
            month=1,
            day=1,
            hour=entry_start.tm_hour,
            minute=entry_start.tm_min,
        ).time(),
        entry_session_end=pd.Timestamp(
            year=2000,
            month=1,
            day=1,
            hour=entry_end.tm_hour,
            minute=entry_end.tm_min,
        ).time(),
    )

    train_rows: list[dict[str, Any]] = []
    for buy_th, sell_th, risk, stop in product(buy_grid, sell_grid, risk_grid, stop_grid):
        pred_df = build_prediction_df(predictor, train_features, buy_th, sell_th)
        config = BacktestConfig(
            initial_capital=args.initial_capital,
            risk_per_trade=risk,
            stop_atr_multiple=stop,
            slippage_pct=args.slippage_pct,
            brokerage_fixed=args.brokerage_fixed,
            brokerage_pct=args.brokerage_pct,
            allow_shorts=True,
        )
        result = Backtester(strategy=strategy, config=config).run(pred_df)
        metrics = result.metrics
        train_rows.append(
            {
                "buy_threshold": buy_th,
                "sell_threshold": sell_th,
                "risk_per_trade": risk,
                "stop_atr_multiple": stop,
                "total_return_pct": float(metrics["total_return_pct"]),
                "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "win_rate": float(metrics["win_rate"]),
                "trades": int(len(result.trades)),
                "score": score_metrics(metrics),
            }
        )

    tune_df = pd.DataFrame(train_rows).sort_values("score", ascending=False).reset_index(drop=True)
    tune_path = out_dir / "train_tuning_results.csv"
    tune_df.to_csv(tune_path, index=False)
    best = tune_df.iloc[0]

    best_buy = float(best["buy_threshold"])
    best_sell = float(best["sell_threshold"])
    best_risk = float(best["risk_per_trade"])
    best_stop = float(best["stop_atr_multiple"])

    oos_pred_df = build_prediction_df(predictor, test_features, best_buy, best_sell)
    oos_pred_path = out_dir / "oos_predictions_best.csv"
    oos_pred_df.to_csv(oos_pred_path, index=False)

    oos_config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=best_risk,
        stop_atr_multiple=best_stop,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
        allow_shorts=True,
    )
    oos_result = Backtester(strategy=strategy, config=oos_config).run(oos_pred_df)
    oos_trades_path = out_dir / "oos_backtest_best_trades.csv"
    oos_equity_path = out_dir / "oos_backtest_best_equity.csv"
    oos_result.trades.to_csv(oos_trades_path, index=False)
    oos_result.equity_curve.to_csv(oos_equity_path, index=False)

    print("OOS Tuning Summary")
    print(f"- Train features rows: {len(train_features)}")
    print(f"- Test features rows: {len(test_features)}")
    print(f"- Holdout classification accuracy: {holdout_metrics['accuracy']:.4f}")
    print(f"- Holdout classification F1 macro: {holdout_metrics['f1_macro']:.4f}")
    print("- Best params from train-window tuning:")
    print(f"  - buy_threshold_proba: {best_buy:.2f}")
    print(f"  - sell_threshold_proba: {best_sell:.2f}")
    print(f"  - risk_per_trade: {best_risk:.4f}")
    print(f"  - stop_atr_multiple: {best_stop:.2f}")
    print(
        f"  - train return %: {float(best['total_return_pct']):.4f}, "
        f"train max_dd %: {float(best['max_drawdown_pct']):.4f}"
    )

    oos_metrics = oos_result.metrics
    print("- OOS backtest with best train params:")
    print(f"  - trades: {len(oos_result.trades)}")
    print(f"  - total return %: {oos_metrics['total_return_pct']:.4f}")
    print(f"  - max drawdown %: {oos_metrics['max_drawdown_pct']:.4f}")
    print(f"  - win rate %: {oos_metrics['win_rate']:.2f}")
    print(f"  - profit factor: {oos_metrics['profit_factor']:.4f}")
    print(f"  - sharpe ratio: {oos_metrics['sharpe_ratio']:.4f}")
    print(f"  - expectancy: {oos_metrics['expectancy']:.4f}")
    print(f"- Train tuning CSV: {tune_path}")
    print(f"- OOS predictions CSV: {oos_pred_path}")
    print(f"- OOS trades CSV: {oos_trades_path}")
    print(f"- OOS equity CSV: {oos_equity_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
