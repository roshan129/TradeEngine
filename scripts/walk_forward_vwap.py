#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import VwapTrendContinuationStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly walk-forward evaluation for the fixed VWAP pullback strategy."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Feature history CSV containing OHLCV + VWAP features",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtests/walk_forward_vwap",
        help="Directory for fold trade/equity files and summary output",
    )
    parser.add_argument("--initial-train-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--entry-start", default="09:20")
    parser.add_argument("--entry-end", default="11:00")
    parser.add_argument("--session-exit", default="15:15")
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument(
        "--exit-mode",
        choices=["rr", "vwap_break", "trailing_low"],
        default="vwap_break",
    )
    parser.add_argument(
        "--fixed-stop-loss-pct",
        type=float,
        default=0.003,
        help="Fixed stop-loss cap below entry as decimal pct (default: 0.003 = 0.30%%)",
    )
    parser.add_argument(
        "--distance-pct",
        type=float,
        default=0.0015,
        help="Minimum close distance above VWAP as decimal pct (default: 0.0015 = 0.15%%)",
    )
    parser.add_argument("--min-candles-above-vwap", type=int, default=5)
    parser.add_argument("--pullback-lookback-bars", type=int, default=5)
    parser.add_argument("--min-pullback-pct", type=float, default=0.0015)
    parser.add_argument("--max-pullback-pct", type=float, default=0.0015)
    parser.add_argument("--vwap-slope-lookback-bars", type=int, default=1)
    parser.add_argument("--min-vwap-slope-pct", type=float, default=0.0001)
    parser.add_argument("--use-ema-filter", action="store_true")
    parser.add_argument("--min-atr-pct", type=float, default=0.0)
    parser.add_argument("--min-bb-width", type=float, default=0.0)
    parser.add_argument("--max-trades-per-day", type=int, default=2)
    parser.add_argument("--stop-after-first-win", action="store_true", default=True)
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Input features contain invalid timestamps")
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    months = _month_sequence(df["timestamp"])
    min_months = args.initial_train_months + args.test_months
    if len(months) < min_months:
        raise ValueError(
            f"Need at least {min_months} months of data, found {len(months)} months."
        )

    fold_idx = 0
    train_end = args.initial_train_months
    summary_rows: list[dict[str, object]] = []
    while train_end + args.test_months <= len(months):
        fold_idx += 1
        train_months = months[train_end - args.initial_train_months : train_end]
        test_months = months[train_end : train_end + args.test_months]

        train_features = df[df["timestamp"].dt.to_period("M").isin(train_months)].reset_index(
            drop=True
        )
        test_features = df[df["timestamp"].dt.to_period("M").isin(test_months)].reset_index(
            drop=True
        )
        if train_features.empty or test_features.empty:
            break

        strategy = VwapTrendContinuationStrategy(
            entry_session_start=_parse_hhmm(args.entry_start),
            entry_session_end=_parse_hhmm(args.entry_end),
            session_exit_time=_parse_hhmm(args.session_exit),
            exit_mode=args.exit_mode,
            risk_reward_multiple=args.rr,
            min_candles_above_vwap=max(1, args.min_candles_above_vwap),
            min_distance_above_vwap_pct=args.distance_pct,
            pullback_lookback_bars=max(1, args.pullback_lookback_bars),
            min_pullback_size_pct=args.min_pullback_pct,
            max_pullback_size_pct=args.max_pullback_pct,
            fixed_stop_loss_pct=args.fixed_stop_loss_pct,
            vwap_slope_lookback_bars=max(1, args.vwap_slope_lookback_bars),
            min_vwap_slope_pct=args.min_vwap_slope_pct,
            min_atr_pct=args.min_atr_pct,
            min_bb_width=args.min_bb_width,
            use_ema_trend_filter=args.use_ema_filter,
            allow_shorts=False,
        )
        result = Backtester(
            strategy=strategy,
            config=BacktestConfig(
                initial_capital=args.initial_capital,
                risk_per_trade=args.risk_per_trade,
                stop_atr_multiple=args.stop_atr_multiple,
                slippage_pct=args.slippage_pct,
                brokerage_fixed=args.brokerage_fixed,
                brokerage_pct=args.brokerage_pct,
                force_end_of_day_exit=True,
                allow_shorts=False,
                max_entries_per_day=args.max_trades_per_day,
                stop_after_first_win_per_day=args.stop_after_first_win,
            ),
        ).run(test_features)

        trades_path = out_dir / f"vwap_fold_{fold_idx}_trades.csv"
        equity_path = out_dir / f"vwap_fold_{fold_idx}_equity.csv"
        result.trades.to_csv(trades_path, index=False)
        result.equity_curve.to_csv(equity_path, index=False)

        summary_rows.append(
            {
                "fold": fold_idx,
                "train_start": str(train_months[0]),
                "train_end": str(train_months[-1]),
                "test_start": str(test_months[0]),
                "test_end": str(test_months[-1]),
                "train_rows": int(len(train_features)),
                "test_rows": int(len(test_features)),
                "trades": int(len(result.trades)),
                "total_return_pct": float(result.metrics["total_return_pct"]),
                "win_rate_pct": float(result.metrics["win_rate"]),
                "profit_factor": float(result.metrics["profit_factor"]),
                "max_drawdown_pct": float(result.metrics["max_drawdown_pct"]),
                "sharpe_ratio": float(result.metrics["sharpe_ratio"]),
                "expectancy": float(result.metrics["expectancy"]),
            }
        )

        print(
            "Fold",
            fold_idx,
            "| train:",
            f"{train_months[0]}->{train_months[-1]}",
            "| test:",
            f"{test_months[0]}->{test_months[-1]}",
            "| trades:",
            len(result.trades),
            "| return %:",
            f"{result.metrics['total_return_pct']:.4f}",
            "| PF:",
            f"{result.metrics['profit_factor']:.4f}",
        )
        train_end += args.step_months

    if not summary_rows:
        print("No folds executed. Check dataset range or window sizes.")
        return 0

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    valid_pf = summary_df["profit_factor"].replace(float("inf"), pd.NA).dropna()
    positive_months = int((summary_df["total_return_pct"] > 0).sum())
    pf_over_one = int((summary_df["profit_factor"] > 1.0).sum())

    print(f"Summary written to: {summary_path}")
    print("Walk-Forward Summary")
    print(f"- Folds: {len(summary_df)}")
    print(f"- Positive months: {positive_months}/{len(summary_df)}")
    print(f"- PF > 1 months: {pf_over_one}/{len(summary_df)}")
    print(f"- Mean return %: {summary_df['total_return_pct'].mean():.4f}")
    print(f"- Median return %: {summary_df['total_return_pct'].median():.4f}")
    print(f"- Mean PF: {valid_pf.mean():.4f}" if not valid_pf.empty else "- Mean PF: n/a")
    print(f"- Median PF: {valid_pf.median():.4f}" if not valid_pf.empty else "- Median PF: n/a")
    print("- Per-fold metrics:")
    print(summary_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
