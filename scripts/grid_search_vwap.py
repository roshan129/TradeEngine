#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from datetime import time
from pathlib import Path

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import VwapTrendContinuationStrategy
from tradeengine.utils.paths import ensure_parent_dir


RR_VALUES = [2.0, 3.0]
SL_VALUES = [0.25, 0.3, 0.35]  # Percent values.
DIST_VALUES = [0.10, 0.15, 0.20]  # Percent values.
ABOVE_VWAP_VALUES = [3, 5, 7]
PULLBACK_VALUES = [0.10, 0.15, 0.20]  # Percent values.
EXIT_MODES = ["rr", "vwap_break", "trailing_low"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search the VWAP pullback strategy on a fixed dataset."
    )
    parser.add_argument(
        "--input",
        default="data/market_data/features/feature_history_last_6months_5m_sbin.csv",
        help="Feature CSV input path (default: SBI 6-month 5-minute feature file)",
    )
    parser.add_argument(
        "--output",
        default="data/backtests/grid_search_results_vwap.csv",
        help="Path for full grid-search results CSV",
    )
    parser.add_argument(
        "--filtered-output",
        default="data/backtests/grid_search_results_vwap_filtered.csv",
        help="Path for filtered/sorted grid-search results CSV",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top rows to print (default: 10)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=20,
        help="Minimum trade count for filtered results (default: 20)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=20.0,
        help="Maximum drawdown pct for filtered results (default: 20.0)",
    )
    parser.add_argument(
        "--entry-start",
        default="09:20",
        help="Entry window start in HH:MM (default: 09:20)",
    )
    parser.add_argument(
        "--entry-end",
        default="11:00",
        help="Entry window end in HH:MM (default: 11:00)",
    )
    parser.add_argument(
        "--session-exit",
        default="15:15",
        help="Session exit time in HH:MM (default: 15:15)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Restrict grid search to long-only mode",
    )
    return parser.parse_args()


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(hour=int(hour), minute=int(minute))


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df


def run_grid_search(
    df: pd.DataFrame,
    entry_start: time,
    entry_end: time,
    session_exit: time,
    long_only: bool,
) -> pd.DataFrame:
    results: list[dict[str, float | int | str | bool]] = []

    for rr, sl_pct, dist_pct, above_vwap_bars, pullback_pct, exit_mode in itertools.product(
        RR_VALUES,
        SL_VALUES,
        DIST_VALUES,
        ABOVE_VWAP_VALUES,
        PULLBACK_VALUES,
        EXIT_MODES,
    ):
        strategy = VwapTrendContinuationStrategy(
            entry_session_start=entry_start,
            entry_session_end=entry_end,
            session_exit_time=session_exit,
            exit_mode=exit_mode,
            risk_reward_multiple=rr,
            min_candles_above_vwap=above_vwap_bars,
            min_distance_above_vwap_pct=dist_pct / 100.0,
            pullback_lookback_bars=5,
            min_pullback_size_pct=pullback_pct / 100.0,
            max_pullback_size_pct=pullback_pct / 100.0,
            fixed_stop_loss_pct=sl_pct / 100.0,
            min_vwap_slope_pct=0.0001,
            allow_shorts=False,
        )
        result = Backtester(
            strategy=strategy,
            config=BacktestConfig(
                initial_capital=100_000.0,
                risk_per_trade=0.01,
                stop_atr_multiple=1.0,
                slippage_pct=0.0005,
                brokerage_fixed=20.0,
                brokerage_pct=0.0003,
                force_end_of_day_exit=True,
                allow_shorts=False,
                max_entries_per_day=2,
                stop_after_first_win_per_day=True,
            ),
        ).run(df)
        metrics = result.metrics
        results.append(
            {
                "rr": rr,
                "sl_pct": sl_pct,
                "dist_pct": dist_pct,
                "min_candles_above_vwap": above_vwap_bars,
                "pullback_pct": pullback_pct,
                "exit_mode": exit_mode,
                "long_only": long_only,
                "return_pct": metrics["total_return_pct"],
                "profit_factor": metrics["profit_factor"],
                "win_rate_pct": metrics["win_rate"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "num_trades": len(result.trades),
                "expectancy": metrics["expectancy"],
                "breakeven_win_rate_pct": metrics["breakeven_win_rate_pct"],
            }
        )

    return pd.DataFrame(results)


def main() -> int:
    args = parse_args()
    df = _load_dataset(args.input)
    results_df = run_grid_search(
        df=df,
        entry_start=_parse_hhmm(args.entry_start),
        entry_end=_parse_hhmm(args.entry_end),
        session_exit=_parse_hhmm(args.session_exit),
        long_only=args.long_only,
    )

    ensure_parent_dir(args.output)
    results_df.to_csv(args.output, index=False)

    filtered_df = results_df[
        (results_df["num_trades"] >= args.min_trades)
        & (results_df["max_drawdown_pct"] < args.max_drawdown)
    ].copy()
    filtered_df = filtered_df.sort_values(
        by=["profit_factor", "max_drawdown_pct", "return_pct"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    ensure_parent_dir(args.filtered_output)
    filtered_df.to_csv(args.filtered_output, index=False)

    print(f"Saved full results: {args.output} (rows={len(results_df)})")
    print(f"Saved filtered results: {args.filtered_output} (rows={len(filtered_df)})")
    print()
    if filtered_df.empty:
        print("No configs survived the filter.")
        return 0

    print(filtered_df.head(args.top).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
