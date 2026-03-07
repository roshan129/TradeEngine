#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
    Strategy,
    VwapRsiMeanReversionStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic candle-by-candle backtest")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing OHLC + indicator columns required by selected strategy",
    )
    parser.add_argument(
        "--trades-output",
        default="backtest_trades.csv",
        help="Output CSV path for trade log (default: backtest_trades.csv)",
    )
    parser.add_argument(
        "--equity-output",
        default="backtest_equity.csv",
        help="Output CSV path for equity curve (default: backtest_equity.csv)",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    parser.add_argument(
        "--strategy",
        choices=[
            "ema_rsi",
            "vwap_rsi_reversion",
            "one_minute_vwap_ema9_scalp",
            "one_minute_vwap_ema9_icici",
        ],
        default="ema_rsi",
        help="Strategy to run (default: ema_rsi)",
    )
    parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Enable short entries/exits for baseline strategy",
    )
    parser.add_argument(
        "--reverse-signals",
        action="store_true",
        help="Reverse directional signals (BUY<->SHORT, SELL<->COVER)",
    )
    parser.add_argument(
        "--scalp-tp-mode",
        choices=["rr", "atr"],
        default="rr",
        help="Take-profit mode for one_minute_vwap_ema9_scalp (default: rr)",
    )
    parser.add_argument(
        "--max-entries-per-day",
        type=int,
        default=0,
        help="Limit number of new entries per day (0 means unlimited)",
    )
    parser.add_argument(
        "--icici-volume-multiplier",
        type=float,
        default=1.8,
        help="Volume spike multiplier for one_minute_vwap_ema9_icici (default: 1.8)",
    )
    parser.add_argument(
        "--icici-risk-reward",
        type=float,
        default=1.5,
        help="Risk-reward multiple for one_minute_vwap_ema9_icici when tp-mode=rr (default: 1.5)",
    )
    parser.add_argument(
        "--icici-session-end",
        default="14:45",
        help="Session end time for one_minute_vwap_ema9_icici in HH:MM (default: 14:45)",
    )
    return parser.parse_args()


def _parse_hhmm(value: str) -> time:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        return time(hour=hour, minute=minute)
    except Exception as exc:  # pragma: no cover - defensive CLI parse
        raise ValueError(f"Invalid HH:MM time value: {value}") from exc


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.input)
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    from_ts = timestamps.min()
    to_ts = timestamps.max()
    short_enabled = args.allow_shorts or args.reverse_signals
    strategy: Strategy
    if args.strategy == "vwap_rsi_reversion":
        strategy = VwapRsiMeanReversionStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "one_minute_vwap_ema9_scalp":
        strategy = OneMinuteVwapEma9ScalpStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
            take_profit_mode=args.scalp_tp_mode,
        )
    elif args.strategy == "one_minute_vwap_ema9_icici":
        strategy = OneMinuteVwapEma9IciciFocusedStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
            take_profit_mode=args.scalp_tp_mode,
            volume_spike_multiplier=args.icici_volume_multiplier,
            risk_reward_multiple=args.icici_risk_reward,
            session_end=_parse_hhmm(args.icici_session_end),
        )
    else:
        strategy = BaselineEmaRsiStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_atr_multiple=args.stop_atr_multiple,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
        allow_shorts=short_enabled,
        max_entries_per_day=(args.max_entries_per_day if args.max_entries_per_day > 0 else None),
    )

    backtester = Backtester(strategy=strategy, config=config)
    result = backtester.run(df)

    result.trades.to_csv(args.trades_output, index=False)
    result.equity_curve.to_csv(args.equity_output, index=False)

    print("Backtest Summary")
    print(f"- Input rows: {len(df)}")
    print(f"- From: {from_ts}")
    print(f"- To: {to_ts}")
    if args.reverse_signals:
        mode = "REVERSED_LONG+SHORT"
    elif short_enabled:
        mode = "LONG+SHORT"
    else:
        mode = "LONG_ONLY"
    print(f"- Strategy: {args.strategy}")
    print(f"- Mode: {mode}")
    print(f"- Total trades: {len(result.trades)}")
    print(f"- Total return %: {result.metrics['total_return_pct']:.4f}")
    print(f"- Win rate %: {result.metrics['win_rate']:.2f}")
    print(f"- Profit factor: {result.metrics['profit_factor']:.4f}")
    print(f"- Gross wins (after cost): {result.metrics['gross_wins_after_cost']:.4f}")
    print(f"- Gross losses (after cost): {result.metrics['gross_losses_after_cost']:.4f}")
    print(f"- Gross wins (before cost): {result.metrics['gross_wins_before_cost']:.4f}")
    print(f"- Gross losses (before cost): {result.metrics['gross_losses_before_cost']:.4f}")
    print(f"- Breakeven win rate %: {result.metrics['breakeven_win_rate_pct']:.2f}")
    print(f"- Max drawdown %: {result.metrics['max_drawdown_pct']:.4f}")
    print(f"- Sharpe ratio: {result.metrics['sharpe_ratio']:.4f}")
    print(f"- Expectancy: {result.metrics['expectancy']:.4f}")
    print(f"- Trades CSV: {args.trades_output}")
    print(f"- Equity CSV: {args.equity_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
