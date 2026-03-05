from __future__ import annotations

import pandas as pd

from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    StrategyContext,
    VwapRsiMeanReversionStrategy,
)


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


def test_baseline_strategy_can_reverse_signals() -> None:
    strategy = BaselineEmaRsiStrategy(allow_shorts=True, reverse_signals=True)
    buy_row = pd.Series({"ema20": 105.0, "ema50": 100.0, "rsi": 60.0})
    long_context = StrategyContext(
        in_position=True,
        available_capital=10_000.0,
        is_end_of_day=False,
        position_side="LONG",
    )

    flat_context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )
    assert strategy.generate_signal(buy_row, flat_context) == "SHORT"
    cover_trigger_row = pd.Series({"ema20": 100.0, "ema50": 99.0, "rsi": 40.0})
    assert strategy.generate_signal(cover_trigger_row, long_context) == "COVER"


def test_vwap_rsi_mean_reversion_signals() -> None:
    strategy = VwapRsiMeanReversionStrategy()
    flat_context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )
    in_position_context = StrategyContext(
        in_position=True,
        available_capital=9_000.0,
        is_end_of_day=False,
        position_side="LONG",
    )

    buy_row = pd.Series({"close": 99.0, "vwap": 100.0, "rsi": 30.0})
    sell_row = pd.Series({"close": 100.0, "vwap": 100.0, "rsi": 50.0})
    hold_row = pd.Series({"close": 101.0, "vwap": 100.0, "rsi": 30.0})

    assert strategy.generate_signal(buy_row, flat_context) == "BUY"
    assert strategy.generate_signal(sell_row, in_position_context) == "SELL"
    assert strategy.generate_signal(hold_row, flat_context) == "HOLD"


def test_vwap_rsi_mean_reversion_stop_loss_uses_1_to_1_rr() -> None:
    strategy = VwapRsiMeanReversionStrategy()
    row = pd.Series({"close": 95.0, "vwap": 100.0})

    stop = strategy.entry_stop_loss(row=row, signal="BUY", stop_atr_multiple=1.0)

    assert stop == 90.0


def test_vwap_rsi_mean_reversion_can_short_and_cover() -> None:
    strategy = VwapRsiMeanReversionStrategy(allow_shorts=True)
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

    short_row = pd.Series({"close": 101.0, "vwap": 100.0, "rsi": 70.0})
    cover_row = pd.Series({"close": 100.0, "vwap": 100.0, "rsi": 45.0})

    assert strategy.generate_signal(short_row, flat_context) == "SHORT"
    assert strategy.generate_signal(cover_row, short_position_context) == "COVER"


def test_vwap_rsi_mean_reversion_can_reverse_signals() -> None:
    strategy = VwapRsiMeanReversionStrategy(allow_shorts=True, reverse_signals=True)
    flat_context = StrategyContext(
        in_position=False,
        available_capital=10_000.0,
        is_end_of_day=False,
    )

    long_setup_row = pd.Series({"close": 99.0, "vwap": 100.0, "rsi": 30.0})
    assert strategy.generate_signal(long_setup_row, flat_context) == "SHORT"
