from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import BaselineEmaRsiStrategy


def _static_backtest_df() -> pd.DataFrame:
    ts = [
        pd.Timestamp("2026-01-01T09:15:00+05:30"),
        pd.Timestamp("2026-01-01T09:20:00+05:30"),
        pd.Timestamp("2026-01-01T09:25:00+05:30"),
        pd.Timestamp("2026-01-01T09:30:00+05:30"),
    ]

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.0, 99.0, 99.0],
            "high": [101.0, 101.0, 100.0, 100.0],
            "low": [80.0, 97.0, 98.0, 98.0],
            "close": [100.0, 99.0, 99.0, 99.0],
            "ema20": [101.0, 101.0, 99.0, 99.0],
            "ema50": [100.0, 100.0, 100.0, 100.0],
            "rsi": [60.0, 60.0, 60.0, 60.0],
            "atr": [2.0, 2.0, 2.0, 2.0],
        }
    )


def test_backtester_stop_loss_only_triggers_from_next_candle() -> None:
    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        stop_atr_multiple=1.0,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
    )
    result = Backtester(strategy=BaselineEmaRsiStrategy(), config=config).run(_static_backtest_df())

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]

    assert trade["entry_timestamp"] == pd.Timestamp("2026-01-01T09:15:00+05:30")
    assert trade["exit_timestamp"] == pd.Timestamp("2026-01-01T09:20:00+05:30")
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["quantity"] == 50
    assert trade["net_pnl"] == pytest.approx(-100.0)
    assert result.equity_curve.iloc[0]["in_position"]


def test_backtester_is_deterministic_for_same_dataset() -> None:
    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        stop_atr_multiple=1.0,
        slippage_pct=0.0005,
        brokerage_fixed=20.0,
        brokerage_pct=0.0003,
    )
    backtester = Backtester(strategy=BaselineEmaRsiStrategy(), config=config)
    df = _static_backtest_df()

    first = backtester.run(df)
    second = backtester.run(df)

    assert first.trades.equals(second.trades)
    assert first.equity_curve.equals(second.equity_curve)
    assert first.metrics == second.metrics
    assert len(first.equity_curve) == len(df)
