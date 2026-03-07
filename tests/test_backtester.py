from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
    VwapRsiMeanReversionStrategy,
)


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


def test_backtester_executes_short_when_enabled() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:20:00+05:30"),
                pd.Timestamp("2026-01-01T09:25:00+05:30"),
            ],
            "open": [100.0, 99.0, 98.0],
            "high": [100.0, 100.0, 99.0],
            "low": [99.0, 98.0, 97.0],
            "close": [99.0, 98.0, 97.0],
            "ema20": [99.0, 99.0, 99.0],
            "ema50": [100.0, 100.0, 100.0],
            "rsi": [40.0, 40.0, 60.0],
            "atr": [2.0, 2.0, 2.0],
        }
    )

    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        stop_atr_multiple=1.0,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=True,
    )

    result = Backtester(
        strategy=BaselineEmaRsiStrategy(allow_shorts=True),
        config=config,
    ).run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["entry_price"] == pytest.approx(99.0)
    assert trade["exit_price"] == pytest.approx(97.0)
    assert trade["net_pnl"] == pytest.approx(100.0)


def test_backtester_reverse_signals_opens_short_on_long_setup() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:20:00+05:30"),
            ],
            "open": [100.0, 99.0],
            "high": [101.0, 99.0],
            "low": [99.0, 98.0],
            "close": [100.0, 98.0],
            "ema20": [101.0, 101.0],
            "ema50": [100.0, 100.0],
            "rsi": [60.0, 60.0],
            "atr": [2.0, 2.0],
        }
    )
    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        stop_atr_multiple=1.0,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=True,
    )
    result = Backtester(
        strategy=BaselineEmaRsiStrategy(allow_shorts=True, reverse_signals=True),
        config=config,
    ).run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["net_pnl"] == pytest.approx(100.0)


def test_backtester_runs_vwap_reversion_strategy() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:20:00+05:30"),
                pd.Timestamp("2026-01-01T09:25:00+05:30"),
            ],
            "open": [95.0, 96.0, 99.0],
            "high": [96.0, 100.0, 101.0],
            "low": [94.0, 95.0, 98.0],
            "close": [95.0, 99.0, 100.0],
            "vwap": [100.0, 100.0, 100.0],
            "rsi": [30.0, 40.0, 45.0],
        }
    )

    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
    )
    result = Backtester(strategy=VwapRsiMeanReversionStrategy(), config=config).run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "LONG"
    assert trade["entry_price"] == pytest.approx(95.0)
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["net_pnl"] == pytest.approx(100.0)


def test_backtester_runs_vwap_reversion_short_trade() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:20:00+05:30"),
                pd.Timestamp("2026-01-01T09:25:00+05:30"),
            ],
            "open": [101.0, 101.0, 100.0],
            "high": [102.0, 101.5, 100.5],
            "low": [100.0, 99.0, 99.0],
            "close": [101.0, 100.0, 100.0],
            "vwap": [100.0, 100.0, 100.0],
            "rsi": [70.0, 60.0, 50.0],
        }
    )

    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=True,
    )
    result = Backtester(
        strategy=VwapRsiMeanReversionStrategy(allow_shorts=True),
        config=config,
    ).run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["entry_price"] == pytest.approx(101.0)
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["net_pnl"] == pytest.approx(99.0)


def test_backtester_runs_vwap_reversion_reversed_mode() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:20:00+05:30"),
            ],
            "open": [99.0, 98.0],
            "high": [100.0, 99.0],
            "low": [98.0, 97.0],
            "close": [99.0, 98.0],
            "vwap": [100.0, 100.0],
            "rsi": [30.0, 30.0],
        }
    )

    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=True,
    )
    result = Backtester(
        strategy=VwapRsiMeanReversionStrategy(allow_shorts=True, reverse_signals=True),
        config=config,
    ).run(df)

    assert len(result.trades) == 1


def test_backtester_runs_one_minute_scalp_strategy() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:15:00+05:30"),
                pd.Timestamp("2026-01-01T09:16:00+05:30"),
                pd.Timestamp("2026-01-01T09:17:00+05:30"),
            ],
            "open": [100.0, 100.2, 100.4],
            "high": [100.8, 100.7, 101.0],
            "low": [99.95, 100.25, 100.35],
            "close": [100.5, 100.45, 100.9],
            "vwap": [100.1, 100.2, 100.2],
            "ema9": [100.2, 100.25, 100.3],
            "volume": [2000.0, 1700.0, 1600.0],
            "rolling_volume_avg": [1000.0, 1000.0, 1000.0],
            "atr": [0.5, 0.5, 0.5],
        }
    )

    config = BacktestConfig(
        initial_capital=100_000.0,
        risk_per_trade=0.01,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=True,
        force_end_of_day_exit=False,
    )
    result = Backtester(
        strategy=OneMinuteVwapEma9ScalpStrategy(allow_shorts=True, take_profit_mode="rr"),
        config=config,
    ).run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "LONG"


def test_backtester_honors_max_entries_per_day_limit() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T09:21:00+05:30"),
                pd.Timestamp("2026-01-01T09:22:00+05:30"),
                pd.Timestamp("2026-01-01T09:23:00+05:30"),
                pd.Timestamp("2026-01-01T09:24:00+05:30"),
            ],
            "open": [100.0, 100.2, 100.1, 100.3],
            "high": [100.5, 100.6, 100.7, 100.8],
            "low": [99.9, 100.0, 100.0, 100.1],
            "close": [100.3, 100.1, 100.5, 100.4],
            "vwap": [100.0, 100.0, 100.0, 100.0],
            "ema9": [100.1, 100.1, 100.1, 100.1],
            "atr": [0.3, 0.3, 0.3, 0.3],
            "bb_width": [0.02, 0.02, 0.02, 0.02],
            "volume": [2500.0, 2500.0, 2500.0, 2500.0],
            "rolling_volume_avg": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        risk_per_trade=0.01,
        slippage_pct=0.0,
        brokerage_fixed=0.0,
        brokerage_pct=0.0,
        allow_shorts=False,
        max_entries_per_day=1,
    )
    result = Backtester(
        strategy=OneMinuteVwapEma9IciciFocusedStrategy(allow_shorts=False, take_profit_mode="rr"),
        config=cfg,
    ).run(df)
    assert len(result.trades) <= 1
