#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import BaselineEmaRsiStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic candle-by-candle backtest")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing OHLC + required indicators (ema20, ema50, rsi, atr)",
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
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.input)
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_atr_multiple=args.stop_atr_multiple,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
    )

    backtester = Backtester(strategy=BaselineEmaRsiStrategy(), config=config)
    result = backtester.run(df)

    result.trades.to_csv(args.trades_output, index=False)
    result.equity_curve.to_csv(args.equity_output, index=False)

    print("Backtest Summary")
    print(f"- Input rows: {len(df)}")
    print(f"- Total trades: {len(result.trades)}")
    print(f"- Total return %: {result.metrics['total_return_pct']:.4f}")
    print(f"- Win rate %: {result.metrics['win_rate']:.2f}")
    print(f"- Profit factor: {result.metrics['profit_factor']:.4f}")
    print(f"- Max drawdown %: {result.metrics['max_drawdown_pct']:.4f}")
    print(f"- Sharpe ratio: {result.metrics['sharpe_ratio']:.4f}")
    print(f"- Expectancy: {result.metrics['expectancy']:.4f}")
    print(f"- Trades CSV: {args.trades_output}")
    print(f"- Equity CSV: {args.equity_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
