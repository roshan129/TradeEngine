from __future__ import annotations

import pandas as pd

from tradeengine.core.strategy import BaselineEmaRsiStrategy, StrategyContext


def test_baseline_strategy_generates_expected_signals() -> None:
    strategy = BaselineEmaRsiStrategy()

    buy_row = pd.Series({"ema20": 105.0, "ema50": 100.0, "rsi": 60.0})
    sell_row = pd.Series({"ema20": 105.0, "ema50": 100.0, "rsi": 40.0})

    flat_context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )
    in_position_context = StrategyContext(
        in_position=True,
        available_capital=9_500.0,
        is_end_of_day=False,
    )

    assert strategy.generate_signal(buy_row, flat_context) == "BUY"
    assert strategy.generate_signal(sell_row, in_position_context) == "SELL"


def test_baseline_strategy_is_deterministic() -> None:
    strategy = BaselineEmaRsiStrategy()
    row = pd.Series({"ema20": 101.0, "ema50": 100.0, "rsi": 58.0})
    context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )

    first = strategy.generate_signal(row, context)
    second = strategy.generate_signal(row, context)

    assert first == second


def test_baseline_strategy_short_signals_when_enabled() -> None:
    strategy = BaselineEmaRsiStrategy(allow_shorts=True)
    short_row = pd.Series({"ema20": 95.0, "ema50": 100.0, "rsi": 40.0})

    flat_context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )
    short_position_context = StrategyContext(
        in_position=True,
        available_capital=10_000.0,
        is_end_of_day=False,
        position_side="SHORT",
    )
    cover_row = pd.Series({"ema20": 95.0, "ema50": 100.0, "rsi": 60.0})

    assert strategy.generate_signal(short_row, flat_context) == "SHORT"
    assert strategy.generate_signal(cover_row, short_position_context) == "COVER"
