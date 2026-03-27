#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time
from itertools import product

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import OneMinuteVwapEma9IciciFocusedStrategy
from tradeengine.utils.paths import ensure_parent_dir


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_time_list(value: str) -> list[time]:
    out: list[time] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        hh, mm = token.split(":", 1)
        out.append(time(hour=int(hh), minute=int(mm)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ICICI 1-minute strategy parameters")
    parser.add_argument("--input", required=True, help="Input 1-minute feature CSV")
    parser.add_argument(
        "--output",
        default="data/backtests/icici_sweep_results.csv",
        help="Results CSV path (default: data/backtests/icici_sweep_results.csv)",
    )
    parser.add_argument("--max-entries", default="4,6,8", help="Comma list: max entries per day")
    parser.add_argument("--volume-multipliers", default="1.6,1.8,2.0")
    parser.add_argument("--risk-rewards", default="1.5,1.8,2.0")
    parser.add_argument("--session-ends", default="14:00,14:30,14:45")
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Run sweep in long-only mode (default: long+short)",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    parser.add_argument("--max-drawdown-filter", type=float, default=2.0)
    parser.add_argument("--profit-factor-filter", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)

    max_entries_list = _parse_int_list(args.max_entries)
    vol_list = _parse_float_list(args.volume_multipliers)
    rr_list = _parse_float_list(args.risk_rewards)
    session_end_list = _parse_time_list(args.session_ends)
    allow_shorts = not args.long_only

    rows: list[dict[str, float | int | str]] = []
    total = len(max_entries_list) * len(vol_list) * len(rr_list) * len(session_end_list)
    done = 0

    for max_entries, vol_mult, rr, session_end in product(
        max_entries_list,
        vol_list,
        rr_list,
        session_end_list,
    ):
        done += 1
        strategy = OneMinuteVwapEma9IciciFocusedStrategy(
            allow_shorts=allow_shorts,
            reverse_signals=False,
            take_profit_mode="rr",
            volume_spike_multiplier=vol_mult,
            risk_reward_multiple=rr,
            session_end=session_end,
        )
        config = BacktestConfig(
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            slippage_pct=args.slippage_pct,
            brokerage_fixed=args.brokerage_fixed,
            brokerage_pct=args.brokerage_pct,
            allow_shorts=allow_shorts,
            max_entries_per_day=max_entries,
        )

        result = Backtester(strategy=strategy, config=config).run(df)
        metrics = result.metrics
        rows.append(
            {
                "max_entries_per_day": max_entries,
                "volume_spike_multiplier": vol_mult,
                "risk_reward_multiple": rr,
                "session_end": session_end.strftime("%H:%M"),
                "trades": int(len(result.trades)),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate": float(metrics["win_rate"]),
                "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
                "expectancy": float(metrics["expectancy"]),
            }
        )
        print(
            f"[{done}/{total}] done: entries={max_entries} "
            f"vol={vol_mult} rr={rr} end={session_end}"
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["total_return_pct", "profit_factor", "win_rate"], ascending=False)
    ensure_parent_dir(args.output)
    out.to_csv(args.output, index=False)

    filtered = out[
        (out["profit_factor"] > args.profit_factor_filter)
        & (out["max_drawdown_pct"] <= args.max_drawdown_filter)
    ]

    print("\nSweep Summary")
    print(f"- Tested combinations: {len(out)}")
    print(f"- Results CSV: {args.output}")
    print(
        f"- Filter: profit_factor > {args.profit_factor_filter}, "
        f"max_drawdown <= {args.max_drawdown_filter}%"
    )
    print(f"- Passing combinations: {len(filtered)}")

    print("\nTop 10 by total_return_pct")
    print(out.head(10).to_string(index=False))

    if not filtered.empty:
        print("\nTop passing combinations")
        print(filtered.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
