#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import InsideBarBreakoutStrategy
from tradeengine.ml.dataset_builder import DatasetBuilder
from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.predictor import ModelPredictor
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly walk-forward: train ML, apply Inside Bar + ML filter on next month."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Feature history CSV containing OHLC + ML feature columns",
    )
    parser.add_argument(
        "--output-dir",
        default="walk_forward_inside_bar_results",
        help="Directory for fold models, predictions, and backtest outputs",
    )
    parser.add_argument("--initial-train-months", type=int, default=8)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--buy-threshold", type=float, default=0.003)
    parser.add_argument("--sell-threshold", type=float, default=-0.003)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--inside-entry-start", default="09:20")
    parser.add_argument("--inside-entry-end", default="10:20")
    parser.add_argument("--inside-max-setup-candles", type=int, default=5)
    parser.add_argument("--inside-min-range-pct", type=float, default=0.0015)
    parser.add_argument("--inside-prob-threshold", type=float, default=0.65)
    parser.add_argument("--inside-use-volume-filter", action="store_true")
    parser.add_argument("--inside-use-vwap-filter", action="store_true")
    parser.add_argument("--inside-use-ema-filter", action="store_true")
    parser.add_argument("--inside-use-atr-stop", action="store_true")
    parser.add_argument("--inside-atr-stop-multiple", type=float, default=1.0)
    parser.add_argument("--inside-rr-multiple", type=float, default=1.0)
    parser.add_argument("--inside-use-inside-range", action="store_true")
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Skip ML training/inference and run pure inside-bar (probability filter disabled)",
    )
    parser.add_argument("--max-trades-per-day", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    return parser.parse_args()


def _parse_hhmm(value: str) -> time:
    hour_text, minute_text = value.split(":", 1)
    return time(hour=int(hour_text), minute=int(minute_text))


def _month_sequence(timestamps: pd.Series) -> list[pd.Period]:
    months = timestamps.dt.to_period("M").dropna().unique().tolist()
    return sorted(months)


def main() -> int:
    args = parse_args()
    if args.initial_train_months <= 0:
        raise ValueError("--initial-train-months must be positive")
    if args.test_months <= 0:
        raise ValueError("--test-months must be positive")
    if args.step_months <= 0:
        raise ValueError("--step-months must be positive")
    if not 0.0 <= args.inside_prob_threshold <= 1.0:
        raise ValueError("--inside-prob-threshold must be in [0, 1]")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Input features contain invalid timestamps")
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    candle_interval_minutes = float(df["timestamp"].diff().median().total_seconds() / 60.0)

    months = _month_sequence(df["timestamp"])
    min_months = args.initial_train_months + args.test_months
    if len(months) < min_months:
        raise ValueError(
            f"Need at least {min_months} months of data, found {len(months)} months."
        )

    horizons = tuple(int(v.strip()) for v in args.horizons.split(",") if v.strip())
    builder = DatasetBuilder()
    available_features = [col for col in ML_FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in ML_FEATURE_COLUMNS if col not in df.columns]
    if not args.no_ml:
        if missing_features:
            print(
                "[WARN] Missing feature columns; training with reduced set: "
                f"{', '.join(available_features)}"
            )
        if not available_features:
            raise ValueError("No ML feature columns available after filtering.")
        trainer = ModelTrainer(feature_columns=available_features, target_column=TARGET_COLUMN)
    else:
        if args.inside_prob_threshold > 0:
            print("[INFO] --no-ml enabled: forcing inside-prob-threshold to 0.0")
        args.inside_prob_threshold = 0.0
        trainer = None

    entry_start = _parse_hhmm(args.inside_entry_start)
    entry_end = _parse_hhmm(args.inside_entry_end)

    fold_idx = 0
    train_end = args.initial_train_months
    summary_rows: list[dict[str, object]] = []
    while train_end + args.test_months <= len(months):
        fold_idx += 1
        train_months = months[:train_end]
        test_months = months[train_end : train_end + args.test_months]

        train_features = df[df["timestamp"].dt.to_period("M").isin(train_months)].reset_index(
            drop=True
        )
        test_features = df[df["timestamp"].dt.to_period("M").isin(test_months)].reset_index(
            drop=True
        )
        if train_features.empty or test_features.empty:
            break

        pred_df = test_features.copy(deep=True)
        if not args.no_ml:
            train_ml = builder.build_dataset(
                train_features,
                horizons=horizons,
                label_horizon=args.label_horizon,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
            )
            assert trainer is not None
            trainer.validate_dataset(train_ml)
            x_train = trainer._prepare_features(train_ml)  # noqa: SLF001
            y_train = train_ml[TARGET_COLUMN].astype(str)

            model = trainer.train(x_train, y_train)
            model_path = out_dir / f"model_fold_{fold_idx}.pkl"
            trainer.save_model(
                model,
                str(model_path),
                metadata={
                    "model_version": "inside_bar_walk_forward",
                    "fold": fold_idx,
                    "train_months": [str(m) for m in train_months],
                    "test_months": [str(m) for m in test_months],
                    "train_rows": int(len(train_ml)),
                    "test_rows": int(len(test_features)),
                    "dataset_path": args.features,
                    "dataset_range_start": str(df["timestamp"].min()),
                    "dataset_range_end": str(df["timestamp"].max()),
                    "candle_interval_minutes": candle_interval_minutes,
                    "feature_count": len(available_features),
                    "missing_features": missing_features,
                },
            )

            predictor = ModelPredictor(str(model_path))
            probabilities = predictor.predict_proba(test_features)
            for column in probabilities.columns:
                pred_df[column] = probabilities[column]

        strategy = InsideBarBreakoutStrategy(
            entry_session_start=entry_start,
            entry_session_end=entry_end,
            max_setup_candles=args.inside_max_setup_candles,
            min_mother_range_pct=args.inside_min_range_pct,
            use_volume_filter=args.inside_use_volume_filter,
            use_vwap_trend_filter=args.inside_use_vwap_filter,
            use_ema_trend_filter=args.inside_use_ema_filter,
            use_atr_stop=args.inside_use_atr_stop,
            atr_stop_multiple=args.inside_atr_stop_multiple,
            risk_reward_multiple=args.inside_rr_multiple,
            use_inside_bar_range=args.inside_use_inside_range,
            probability_threshold=args.inside_prob_threshold,
            allow_shorts=args.allow_shorts,
        )
        bt_config = BacktestConfig(
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            stop_atr_multiple=args.stop_atr_multiple,
            slippage_pct=args.slippage_pct,
            brokerage_fixed=args.brokerage_fixed,
            brokerage_pct=args.brokerage_pct,
            allow_shorts=args.allow_shorts,
            max_entries_per_day=args.max_trades_per_day,
        )
        result = Backtester(strategy=strategy, config=bt_config).run(pred_df)

        trades_path = out_dir / f"inside_bar_fold_{fold_idx}_trades.csv"
        equity_path = out_dir / f"inside_bar_fold_{fold_idx}_equity.csv"
        pred_path = out_dir / f"inside_bar_fold_{fold_idx}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        result.trades.to_csv(trades_path, index=False)
        result.equity_curve.to_csv(equity_path, index=False)

        summary_rows.append(
            {
                "fold": fold_idx,
                "train_months": len(train_months),
                "test_months": len(test_months),
                "test_start": str(test_months[0]),
                "test_end": str(test_months[-1]),
                "trades": len(result.trades),
                "total_return_pct": result.metrics["total_return_pct"],
                "win_rate_pct": result.metrics["win_rate"],
                "profit_factor": result.metrics["profit_factor"],
                "max_drawdown_pct": result.metrics["max_drawdown_pct"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
            }
        )

        print(
            "Fold",
            fold_idx,
            "| train months:",
            len(train_months),
            "| test months:",
            len(test_months),
            "| trades:",
            len(result.trades),
            "| return %:",
            f"{result.metrics['total_return_pct']:.4f}",
            "| win rate %:",
            f"{result.metrics['win_rate']:.2f}",
        )

        train_end += args.step_months

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "walk_forward_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary written to: {summary_path}")
    else:
        print("No folds executed. Check data range or window sizes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
