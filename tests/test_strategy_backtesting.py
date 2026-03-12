from __future__ import annotations

import pandas as pd

from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    MLSignalStrategy,
    OpeningRangeBreakoutStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
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


def test_one_minute_scalp_strategy_entry_and_tp_exit() -> None:
    strategy = OneMinuteVwapEma9ScalpStrategy(allow_shorts=True, take_profit_mode="rr")
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    entry_row = pd.Series(
        {
            "open": 100.0,
            "high": 100.8,
            "low": 99.95,
            "close": 100.5,
            "vwap": 100.1,
            "ema9": 100.2,
            "volume": 1800.0,
            "rolling_volume_avg": 1000.0,
            "atr": 0.5,
        }
    )
    assert strategy.generate_signal(entry_row, flat_context) == "BUY"

    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.0,
        position_stop_loss=99.75,
    )
    tp_row = pd.Series(
        {
            "open": 100.3,
            "high": 100.7,
            "low": 100.2,
            "close": 100.4,
            "vwap": 100.2,
            "ema9": 100.3,
            "volume": 1500.0,
            "rolling_volume_avg": 1000.0,
            "atr": 0.5,
        }
    )
    # Risk=0.25; RR target=100 + 1.2*0.25 = 100.3 => SELL
    assert strategy.generate_signal(tp_row, in_pos_context) == "SELL"


def test_one_minute_icici_strategy_respects_session_filter() -> None:
    strategy = OneMinuteVwapEma9IciciFocusedStrategy()
    context = StrategyContext(in_position=False, available_capital=100_000.0, is_end_of_day=False)
    out_of_session_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T09:16:00+05:30"),
            "open": 100.0,
            "high": 100.4,
            "low": 99.9,
            "close": 100.2,
            "vwap": 100.0,
            "ema9": 100.1,
            "atr": 0.3,
            "bb_width": 0.01,
            "volume": 2500.0,
            "rolling_volume_avg": 1000.0,
        }
    )
    assert strategy.generate_signal(out_of_session_row, context) == "HOLD"


def test_ml_signal_strategy_maps_prediction_column() -> None:
    strategy = MLSignalStrategy()
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=95_000.0,
        is_end_of_day=False,
        position_side="LONG",
    )

    assert strategy.generate_signal(pd.Series({"prediction": "BUY"}), flat_context) == "BUY"
    assert strategy.generate_signal(pd.Series({"prediction": "SELL"}), flat_context) == "HOLD"
    assert strategy.generate_signal(pd.Series({"prediction": "SELL"}), in_pos_context) == "SELL"


def test_ml_signal_strategy_blocks_entries_outside_entry_window() -> None:
    strategy = MLSignalStrategy()
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=95_000.0,
        is_end_of_day=False,
        position_side="LONG",
    )
    out_of_window_buy = pd.Series(
        {"timestamp": pd.Timestamp("2026-01-01T11:00:00+05:30"), "prediction": "BUY"}
    )
    out_of_window_sell = pd.Series(
        {"timestamp": pd.Timestamp("2026-01-01T11:00:00+05:30"), "prediction": "SELL"}
    )

    assert strategy.generate_signal(out_of_window_buy, flat_context) == "HOLD"
    assert strategy.generate_signal(out_of_window_sell, in_pos_context) == "SELL"


def test_opening_range_breakout_requires_opening_window_complete() -> None:
    strategy = OpeningRangeBreakoutStrategy(probability_threshold=0.65)
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )

    opening_915 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-02T09:15:00+05:30"),
            "high": 100.0,
            "low": 99.0,
        }
    )
    opening_920 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-02T09:20:00+05:30"),
            "high": 101.0,
            "low": 98.5,
        }
    )
    breakout = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-02T09:21:00+05:30"),
            "high": 101.5,
            "low": 100.5,
            "buy_probability": 0.7,
            "sell_probability": 0.1,
        }
    )

    assert strategy.generate_signal(opening_915, flat_context) == "HOLD"
    assert strategy.generate_signal(opening_920, flat_context) == "HOLD"
    assert strategy.generate_signal(breakout, flat_context) == "BUY"


def test_opening_range_breakout_respects_probability_threshold_and_tp() -> None:
    strategy = OpeningRangeBreakoutStrategy(probability_threshold=0.65)
    blocked_breakout = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-03T09:21:00+05:30"),
            "high": 101.2,
            "low": 100.0,
            "buy_probability": 0.6,
            "sell_probability": 0.1,
        }
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )

    # Seed opening range for the day.
    strategy.generate_signal(
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-03T09:15:00+05:30"),
                "high": 100.0,
                "low": 99.0,
            }
        ),
        flat_context,
    )
    strategy.generate_signal(
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-03T09:20:00+05:30"),
                "high": 100.5,
                "low": 98.8,
            }
        ),
        flat_context,
    )

    assert strategy.generate_signal(blocked_breakout, flat_context) == "HOLD"

    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.0,
        position_stop_loss=99.75,
    )
    tp_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-03T09:30:00+05:30"),
            "high": 100.3,
            "low": 99.9,
        }
    )
    assert strategy.generate_signal(tp_row, in_pos_context) == "SELL"
