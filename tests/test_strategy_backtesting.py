from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    FirstFiveMinuteCandleMomentumStrategy,
    FirstFiveMinuteFakeBreakoutStrategy,
    InsideBarBreakoutStrategy,
    InsideBarFakeBreakoutReversalStrategy,
    MLSignalStrategy,
    OpeningRangeBreakoutStrategy,
    OpeningRangePullbackStrategy,
    RandomOpenDirectionStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
    SupportResistanceReversalStrategy,
    StrategyContext,
    VwapTrendContinuationStrategy,
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


def test_opening_range_pullback_waits_for_breakout_then_retest_entry() -> None:
    strategy = OpeningRangePullbackStrategy()
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )

    opening_915 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:15:00+05:30"),
            "high": 100.0,
            "low": 99.5,
            "close": 99.8,
        }
    )
    opening_920 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:20:00+05:30"),
            "high": 100.5,
            "low": 99.7,
            "close": 100.2,
        }
    )
    opening_925 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:25:00+05:30"),
            "high": 100.8,
            "low": 100.1,
            "close": 100.6,
        }
    )
    breakout_926 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:26:00+05:30"),
            "high": 101.3,
            "low": 100.9,
            "close": 101.0,
        }
    )
    pullback_927 = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:27:00+05:30"),
            "high": 100.95,
            "low": 100.75,
            "close": 100.85,
        }
    )

    assert strategy.generate_signal(opening_915, flat_context) == "HOLD"
    assert strategy.generate_signal(opening_920, flat_context) == "HOLD"
    assert strategy.generate_signal(opening_925, flat_context) == "HOLD"
    assert strategy.generate_signal(breakout_926, flat_context) == "HOLD"
    assert strategy.generate_signal(pullback_927, flat_context) == "BUY"

    stop = strategy.entry_stop_loss(pullback_927, signal="BUY", stop_atr_multiple=1.0)
    assert stop == pytest.approx(99.5)


def test_opening_range_pullback_exits_at_r_multiple_target() -> None:
    strategy = OpeningRangePullbackStrategy(risk_reward_multiple=1.5)
    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.8,
        position_stop_loss=99.8,
    )
    tp_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:35:00+05:30"),
            "high": 102.4,
            "low": 100.9,
        }
    )

    assert strategy.generate_signal(tp_row, in_pos_context) == "SELL"


def test_vwap_trend_continuation_buys_after_shallow_pullback_breakout() -> None:
    strategy = VwapTrendContinuationStrategy(pullback_lookback_bars=3)
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-07T09:20:00+05:30"),
                "open": 100.20,
                "high": 100.80,
                "low": 100.35,
                "close": 100.70,
                "vwap": 100.00,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-07T09:25:00+05:30"),
                "open": 100.72,
                "high": 101.00,
                "low": 100.55,
                "close": 100.90,
                "vwap": 100.10,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"),
                "open": 100.92,
                "high": 101.20,
                "low": 100.75,
                "close": 101.10,
                "vwap": 100.20,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-07T09:35:00+05:30"),
                "open": 101.10,
                "high": 101.14,
                "low": 101.02,
                "close": 101.05,
                "vwap": 100.40,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"),
                "open": 101.06,
                "high": 101.18,
                "low": 101.03,
                "close": 101.14,
                "vwap": 100.50,
            }
        ),
    ]

    for row in rows[:-1]:
        assert strategy.generate_signal(row, flat_context) == "HOLD"
    assert strategy.generate_signal(rows[-1], flat_context) == "BUY"

    stop = strategy.entry_stop_loss(rows[-1], signal="BUY", stop_atr_multiple=1.0)
    assert stop == pytest.approx(101.02)


def test_vwap_trend_continuation_caps_stop_with_fixed_percent() -> None:
    strategy = VwapTrendContinuationStrategy(
        pullback_lookback_bars=3,
        max_pullback_size_pct=0.006,
        fixed_stop_loss_pct=0.003,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:20:00+05:30"), "open": 100.2, "high": 100.8, "low": 100.35, "close": 100.7, "vwap": 100.0}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:25:00+05:30"), "open": 100.72, "high": 101.0, "low": 100.55, "close": 100.9, "vwap": 100.1}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"), "open": 100.92, "high": 101.2, "low": 100.75, "close": 101.1, "vwap": 100.2}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:35:00+05:30"), "open": 101.1, "high": 101.14, "low": 100.70, "close": 101.0, "vwap": 100.35}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"), "open": 101.0, "high": 101.18, "low": 100.95, "close": 101.10, "vwap": 100.45}),
    ]

    for row in rows[:-1]:
        assert strategy.generate_signal(row, flat_context) == "HOLD"
    assert strategy.generate_signal(rows[-1], flat_context) == "BUY"

    stop = strategy.entry_stop_loss(rows[-1], signal="BUY", stop_atr_multiple=1.0)
    assert stop == pytest.approx(101.10 * (1.0 - 0.003))


def test_vwap_trend_continuation_requires_recent_separation_from_vwap() -> None:
    strategy = VwapTrendContinuationStrategy(pullback_lookback_bars=3)
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:20:00+05:30"), "open": 100.2, "high": 100.8, "low": 100.35, "close": 100.7, "vwap": 100.0}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:25:00+05:30"), "open": 100.72, "high": 101.0, "low": 100.10, "close": 100.9, "vwap": 100.1}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"), "open": 100.92, "high": 101.2, "low": 100.75, "close": 101.1, "vwap": 100.2}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:35:00+05:30"), "open": 101.1, "high": 101.14, "low": 101.02, "close": 101.05, "vwap": 100.4}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"), "open": 101.06, "high": 101.18, "low": 101.03, "close": 101.14, "vwap": 100.5}),
    ]

    for row in rows:
        assert strategy.generate_signal(row, flat_context) == "HOLD"


def test_vwap_trend_continuation_skips_flat_vwap() -> None:
    strategy = VwapTrendContinuationStrategy(
        pullback_lookback_bars=3,
        min_vwap_slope_pct=0.002,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:20:00+05:30"), "open": 100.2, "high": 100.8, "low": 100.35, "close": 100.7, "vwap": 100.0}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:25:00+05:30"), "open": 100.72, "high": 101.0, "low": 100.55, "close": 100.9, "vwap": 100.05}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"), "open": 100.92, "high": 101.2, "low": 100.75, "close": 101.1, "vwap": 100.08}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:35:00+05:30"), "open": 101.1, "high": 101.14, "low": 101.02, "close": 101.05, "vwap": 100.10}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"), "open": 101.06, "high": 101.18, "low": 101.03, "close": 101.14, "vwap": 100.12}),
    ]

    for row in rows:
        assert strategy.generate_signal(row, flat_context) == "HOLD"


def test_vwap_trend_continuation_respects_ema_and_regime_filters() -> None:
    strategy = VwapTrendContinuationStrategy(
        pullback_lookback_bars=3,
        use_ema_trend_filter=True,
        min_atr_pct=0.003,
        min_bb_width=0.01,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    seed_rows = [
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:20:00+05:30"), "open": 100.2, "high": 100.8, "low": 100.35, "close": 100.7, "vwap": 100.0, "ema20": 100.3, "ema50": 100.1, "atr": 0.45, "bb_width": 0.02}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:25:00+05:30"), "open": 100.72, "high": 101.0, "low": 100.55, "close": 100.9, "vwap": 100.1, "ema20": 100.45, "ema50": 100.2, "atr": 0.45, "bb_width": 0.02}),
        pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"), "open": 100.92, "high": 101.2, "low": 100.75, "close": 101.1, "vwap": 100.2, "ema20": 100.6, "ema50": 100.3, "atr": 0.45, "bb_width": 0.02}),
    ]
    blocked_setup = pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:35:00+05:30"), "open": 101.1, "high": 101.14, "low": 101.02, "close": 101.05, "vwap": 100.4, "ema20": 100.2, "ema50": 100.3, "atr": 0.10, "bb_width": 0.005})
    blocked_trigger = pd.Series({"timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"), "open": 101.06, "high": 101.18, "low": 101.03, "close": 101.14, "vwap": 100.5, "ema20": 100.3, "ema50": 100.4, "atr": 0.10, "bb_width": 0.005})

    for row in seed_rows:
        assert strategy.generate_signal(row, flat_context) == "HOLD"
    assert strategy.generate_signal(blocked_setup, flat_context) == "HOLD"
    assert strategy.generate_signal(blocked_trigger, flat_context) == "HOLD"


def test_vwap_trend_continuation_uses_rr_target_for_exit() -> None:
    strategy = VwapTrendContinuationStrategy(exit_mode="rr", risk_reward_multiple=2.0)
    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=101.0,
        position_stop_loss=100.7,
    )
    target_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:45:00+05:30"),
            "open": 101.1,
            "high": 101.7,
            "low": 101.0,
            "close": 101.4,
            "vwap": 100.6,
        }
    )

    assert strategy.generate_signal(target_row, in_pos_context) == "SELL"


def test_vwap_trend_continuation_can_exit_on_vwap_break() -> None:
    strategy = VwapTrendContinuationStrategy(exit_mode="vwap_break")
    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=101.0,
        position_stop_loss=100.7,
    )
    exit_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:45:00+05:30"),
            "open": 101.1,
            "high": 101.2,
            "low": 100.2,
            "close": 100.4,
            "vwap": 100.5,
        }
    )

    assert strategy.generate_signal(exit_row, in_pos_context) == "SELL"


def test_vwap_trend_continuation_can_exit_on_trailing_structure_break() -> None:
    strategy = VwapTrendContinuationStrategy(exit_mode="trailing_low")
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    seed_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:40:00+05:30"),
            "open": 101.0,
            "high": 101.4,
            "low": 101.1,
            "close": 101.3,
            "vwap": 100.5,
        }
    )
    assert strategy.generate_signal(seed_row, flat_context) == "HOLD"

    in_pos_context = StrategyContext(
        in_position=True,
        available_capital=90_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=101.0,
        position_stop_loss=100.7,
    )
    exit_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:45:00+05:30"),
            "open": 101.2,
            "high": 101.25,
            "low": 100.9,
            "close": 101.0,
            "vwap": 100.6,
        }
    )

    assert strategy.generate_signal(exit_row, in_pos_context) == "SELL"


def test_support_resistance_reversal_can_trade_external_support_level() -> None:
    strategy = SupportResistanceReversalStrategy(
        use_external_levels=True,
        allow_shorts=True,
        use_trend_filter=False,
        distance_threshold_pct=0.003,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:20:00+05:30"),
            "open": 99.9,
            "high": 100.2,
            "low": 99.6,
            "close": 100.1,
            "volume": 1500.0,
            "rolling_volume_avg": 1000.0,
            "vwap": 99.9,
            "ema20": 100.0,
            "sr_support_price": 99.8,
            "sr_support_touch_count": 3,
            "sr_resistance_price": 100.8,
            "sr_resistance_touch_count": 2,
        }
    )

    assert strategy.generate_signal(row, flat_context) == "BUY"
    assert strategy.entry_stop_loss(row=row, signal="BUY", stop_atr_multiple=1.0) == pytest.approx(
        99.7002
    )


def test_support_resistance_reversal_external_level_respects_cooldown() -> None:
    strategy = SupportResistanceReversalStrategy(
        use_external_levels=True,
        cooldown_candles=10,
        allow_shorts=True,
        use_trend_filter=False,
        distance_threshold_pct=0.003,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:20:00+05:30"),
            "open": 99.9,
            "high": 100.2,
            "low": 99.6,
            "close": 100.1,
            "volume": 1500.0,
            "rolling_volume_avg": 1000.0,
            "vwap": 99.9,
            "ema20": 100.0,
            "sr_support_price": 99.8,
            "sr_support_touch_count": 3,
            "sr_resistance_price": 100.8,
            "sr_resistance_touch_count": 2,
        }
    )

    assert strategy.generate_signal(row, flat_context) == "BUY"
    strategy.entry_stop_loss(row=row, signal="BUY", stop_atr_multiple=1.0)

    next_row = row.copy()
    next_row["timestamp"] = pd.Timestamp("2026-01-05T09:21:00+05:30")
    assert strategy.generate_signal(next_row, flat_context) == "HOLD"


def test_random_open_direction_is_deterministic_for_same_day_and_seed() -> None:
    strategy_a = RandomOpenDirectionStrategy(seed=7, allow_shorts=True)
    strategy_b = RandomOpenDirectionStrategy(seed=7, allow_shorts=True)
    context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:15:00+05:30"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        }
    )

    assert strategy_a.generate_signal(row, context) == strategy_b.generate_signal(row, context)


def test_random_open_direction_uses_candle_extreme_as_stop() -> None:
    strategy = RandomOpenDirectionStrategy(seed=7, allow_shorts=True)
    context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-05T09:15:00+05:30"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        }
    )

    signal = strategy.generate_signal(row, context)
    stop = strategy.entry_stop_loss(row=row, signal=signal, stop_atr_multiple=1.0)
    if signal == "BUY":
        assert stop == 99.0
    else:
        assert stop == 101.0


def test_first_five_minute_momentum_triggers_bullish_breakout() -> None:
    strategy = FirstFiveMinuteCandleMomentumStrategy(
        allow_shorts=True,
        use_gap_filter=False,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    opening_rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-06T09:15:00+05:30"),
                "open": 100.0,
                "high": 100.4,
                "low": 99.9,
                "close": 100.3,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.0,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-06T09:16:00+05:30"),
                "open": 100.3,
                "high": 100.6,
                "low": 100.2,
                "close": 100.5,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.1,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-06T09:17:00+05:30"),
                "open": 100.5,
                "high": 100.8,
                "low": 100.4,
                "close": 100.7,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.2,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-06T09:18:00+05:30"),
                "open": 100.7,
                "high": 101.0,
                "low": 100.6,
                "close": 100.9,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.4,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-06T09:19:00+05:30"),
                "open": 100.9,
                "high": 101.2,
                "low": 100.8,
                "close": 101.1,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.6,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        ),
    ]
    for row in opening_rows:
        assert strategy.generate_signal(row, flat_context) == "HOLD"

    breakout_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-06T09:20:00+05:30"),
            "open": 101.1,
            "high": 101.3,
            "low": 101.0,
            "close": 101.25,
            "volume": 1500.0,
            "rolling_volume_avg": 900.0,
            "vwap": 100.8,
            "atr": 0.5,
            "gap_percent": 0.4,
        }
    )

    assert strategy.generate_signal(breakout_row, flat_context) == "BUY"
    assert strategy.entry_stop_loss(
        row=breakout_row, signal="BUY", stop_atr_multiple=1.0
    ) == pytest.approx(99.9)


def test_first_five_minute_momentum_reverse_sets_short_stop_correctly() -> None:
    strategy = FirstFiveMinuteCandleMomentumStrategy(
        allow_shorts=True,
        reverse_signals=True,
        use_gap_filter=False,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    for minute, high, low, close in [
        (15, 100.4, 99.9, 100.3),
        (16, 100.6, 100.2, 100.5),
        (17, 100.8, 100.4, 100.7),
        (18, 101.0, 100.6, 100.9),
        (19, 101.2, 100.8, 101.1),
    ]:
        row = pd.Series(
            {
                "timestamp": pd.Timestamp(f"2026-01-06T09:{minute}:00+05:30"),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.0,
                "atr": 0.5,
                "gap_percent": 0.4,
            }
        )
        strategy.generate_signal(row, flat_context)

    breakout_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-06T09:20:00+05:30"),
            "open": 101.1,
            "high": 101.3,
            "low": 101.0,
            "close": 101.25,
            "volume": 1500.0,
            "rolling_volume_avg": 900.0,
            "vwap": 100.8,
            "atr": 0.5,
            "gap_percent": 0.4,
        }
    )

    assert strategy.generate_signal(breakout_row, flat_context) == "SHORT"
    assert strategy.entry_stop_loss(
        row=breakout_row, signal="SHORT", stop_atr_multiple=1.0
    ) == pytest.approx(101.2)


def test_first_five_minute_fake_breakout_triggers_short_after_failed_up_break() -> None:
    strategy = FirstFiveMinuteFakeBreakoutStrategy(allow_shorts=True)
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    for minute, high, low in [
        (15, 100.4, 99.9),
        (16, 100.6, 100.2),
        (17, 100.8, 100.4),
        (18, 101.0, 100.6),
        (19, 101.2, 100.8),
    ]:
        row = pd.Series(
            {
                "timestamp": pd.Timestamp(f"2026-01-07T09:{minute}:00+05:30"),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": high - 0.1,
                "volume": 1000.0,
                "rolling_volume_avg": 900.0,
                "vwap": 100.0,
            }
        )
        assert strategy.generate_signal(row, flat_context) == "HOLD"

    breakout_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:22:00+05:30"),
            "open": 101.15,
            "high": 101.5,
            "low": 101.1,
            "close": 101.4,
            "volume": 1200.0,
            "rolling_volume_avg": 900.0,
            "vwap": 101.0,
        }
    )
    assert strategy.generate_signal(breakout_row, flat_context) == "HOLD"

    failure_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:24:00+05:30"),
            "open": 101.25,
            "high": 101.3,
            "low": 100.9,
            "close": 101.0,
            "volume": 1400.0,
            "rolling_volume_avg": 900.0,
            "vwap": 101.2,
        }
    )
    assert strategy.generate_signal(failure_row, flat_context) == "SHORT"
    assert strategy.entry_stop_loss(
        row=failure_row, signal="SHORT", stop_atr_multiple=1.0
    ) == pytest.approx(101.5)


def test_first_five_minute_momentum_uses_fixed_percent_stop_loss() -> None:
    strategy = FirstFiveMinuteCandleMomentumStrategy(
        fixed_stop_loss_pct=0.0025,
        use_gap_filter=False,
    )
    row = pd.Series({"close": 100.0})

    assert strategy.entry_stop_loss(row=row, signal="BUY", stop_atr_multiple=1.0) == pytest.approx(
        99.75
    )


def test_first_five_minute_fake_breakout_uses_fixed_percent_take_profit() -> None:
    strategy = FirstFiveMinuteFakeBreakoutStrategy(
        fixed_take_profit_pct=0.005,
    )
    context = StrategyContext(
        in_position=True,
        available_capital=100_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.0,
        position_stop_loss=99.0,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-07T09:30:00+05:30"),
            "high": 100.6,
            "low": 100.1,
            "close": 100.4,
            "volume": 1500.0,
            "rolling_volume_avg": 900.0,
            "vwap": 100.2,
        }
    )

    assert strategy.generate_signal(row, context) == "SELL"


def test_inside_bar_breakout_can_filter_long_entries_by_distance_from_open() -> None:
    strategy = InsideBarBreakoutStrategy(
        allow_longs=True,
        allow_shorts=False,
        use_volume_filter=False,
        min_mother_range_pct=0.0,
        min_distance_from_open=2.0,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )

    setup_rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-08T09:30:00+05:30"),
                "high": 100.0,
                "low": 95.0,
                "close": 98.0,
                "volume": 1000.0,
                "distance_from_open": 1.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-08T09:45:00+05:30"),
                "high": 99.0,
                "low": 96.0,
                "close": 98.5,
                "volume": 1000.0,
                "distance_from_open": 1.5,
            }
        ),
    ]
    for row in setup_rows:
        assert strategy.generate_signal(row, flat_context) == "HOLD"

    blocked_breakout = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-08T10:00:00+05:30"),
            "high": 101.0,
            "low": 98.0,
            "close": 100.5,
            "volume": 1000.0,
            "distance_from_open": 1.8,
        }
    )
    assert strategy.generate_signal(blocked_breakout, flat_context) == "HOLD"

    strategy = InsideBarBreakoutStrategy(
        allow_longs=True,
        allow_shorts=False,
        use_volume_filter=False,
        min_mother_range_pct=0.0,
        min_distance_from_open=2.0,
    )
    for row in setup_rows:
        strategy.generate_signal(row, flat_context)
    allowed_breakout = blocked_breakout.copy()
    allowed_breakout["distance_from_open"] = 2.5
    assert strategy.generate_signal(allowed_breakout, flat_context) == "BUY"


def test_inside_bar_breakout_without_setup_expiry_keeps_setup_active() -> None:
    strategy = InsideBarBreakoutStrategy(
        allow_longs=True,
        allow_shorts=False,
        use_volume_filter=False,
        min_mother_range_pct=0.0,
        max_setup_candles=0,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )

    rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-09T09:15:00+05:30"),
                "high": 100.0,
                "low": 95.0,
                "close": 98.0,
                "volume": 1000.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-09T09:30:00+05:30"),
                "high": 99.0,
                "low": 96.0,
                "close": 98.5,
                "volume": 1000.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-09T09:45:00+05:30"),
                "high": 98.5,
                "low": 96.5,
                "close": 98.0,
                "volume": 1000.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-09T10:00:00+05:30"),
                "high": 98.0,
                "low": 96.8,
                "close": 97.5,
                "volume": 1000.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-09T10:15:00+05:30"),
                "high": 101.0,
                "low": 97.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        ),
    ]

    for row in rows[:-1]:
        assert strategy.generate_signal(row, flat_context) == "HOLD"
    assert strategy.generate_signal(rows[-1], flat_context) == "BUY"


def test_inside_bar_breakout_can_hold_after_entry_cutoff_until_session_exit() -> None:
    strategy = InsideBarBreakoutStrategy(
        entry_session_end=pd.Timestamp("2026-01-01T14:15:00+05:30").time(),
        session_exit_time=pd.Timestamp("2026-01-01T15:15:00+05:30").time(),
    )
    context = StrategyContext(
        in_position=True,
        available_capital=100_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.0,
        position_stop_loss=99.0,
    )
    before_exit_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T14:30:00+05:30"),
            "high": 100.4,
            "low": 99.8,
            "close": 100.1,
            "volume": 1000.0,
        }
    )
    at_exit_row = before_exit_row.copy()
    at_exit_row["timestamp"] = pd.Timestamp("2026-01-01T15:15:00+05:30")

    assert strategy.generate_signal(before_exit_row, context) == "HOLD"
    assert strategy.generate_signal(at_exit_row, context) == "SELL"


def test_inside_bar_breakout_can_cap_stop_loss_distance() -> None:
    strategy = InsideBarBreakoutStrategy(
        max_stop_loss_pct=0.007,
        use_volume_filter=False,
        min_mother_range_pct=0.0,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T10:00:00+05:30"),
            "close": 100.0,
        }
    )
    strategy._mother_high = 101.2
    strategy._mother_low = 99.0

    long_stop = strategy.entry_stop_loss(row=row, signal="BUY", stop_atr_multiple=1.0)
    short_stop = strategy.entry_stop_loss(row=row, signal="SHORT", stop_atr_multiple=1.0)

    assert long_stop == pytest.approx(99.3)
    assert short_stop == pytest.approx(100.7)


def test_inside_bar_breakout_can_exit_after_five_candles_without_profit() -> None:
    strategy = InsideBarBreakoutStrategy(
        max_holding_candles_without_profit=5,
        session_exit_time=pd.Timestamp("2026-01-01T15:15:00+05:30").time(),
    )
    strategy._current_day = pd.Timestamp("2026-01-01T10:30:00+05:30").date()
    strategy._position_entry_bar = 1
    strategy._bar_index = 5
    context = StrategyContext(
        in_position=True,
        available_capital=100_000.0,
        is_end_of_day=False,
        position_side="LONG",
        position_entry_price=100.0,
        position_stop_loss=99.0,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T10:30:00+05:30"),
            "high": 100.2,
            "low": 99.7,
            "close": 99.9,
            "volume": 1000.0,
        }
    )

    assert strategy.generate_signal(row, context) == "SELL"


def test_inside_bar_breakout_keeps_position_if_profitable_after_five_candles() -> None:
    strategy = InsideBarBreakoutStrategy(
        max_holding_candles_without_profit=5,
        risk_reward_multiple=2.0,
        session_exit_time=pd.Timestamp("2026-01-01T15:15:00+05:30").time(),
    )
    strategy._current_day = pd.Timestamp("2026-01-01T10:30:00+05:30").date()
    strategy._position_entry_bar = 1
    strategy._bar_index = 5
    context = StrategyContext(
        in_position=True,
        available_capital=100_000.0,
        is_end_of_day=False,
        position_side="SHORT",
        position_entry_price=100.0,
        position_stop_loss=101.0,
    )
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T10:30:00+05:30"),
            "high": 100.1,
            "low": 99.4,
            "close": 99.8,
            "volume": 1000.0,
        }
    )

    assert strategy.generate_signal(row, context) == "HOLD"


def test_inside_bar_fake_breakout_reversal_can_short_after_failed_upside_breakout() -> None:
    strategy = InsideBarFakeBreakoutReversalStrategy(
        allow_longs=False,
        allow_shorts=True,
        max_stop_loss_pct=0.006,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-01T09:15:00+05:30"),
                "high": 100.0,
                "low": 95.0,
                "close": 98.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-01T09:30:00+05:30"),
                "high": 99.0,
                "low": 96.0,
                "close": 98.5,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-01T09:45:00+05:30"),
                "high": 101.0,
                "low": 98.0,
                "close": 100.8,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-01T10:00:00+05:30"),
                "high": 100.2,
                "low": 97.5,
                "close": 99.8,
            }
        ),
    ]

    for row in rows[:-1]:
        assert strategy.generate_signal(row, flat_context) == "HOLD"

    assert strategy.generate_signal(rows[-1], flat_context) == "SHORT"
    stop = strategy.entry_stop_loss(rows[-1], "SHORT", 1.0)
    assert stop == pytest.approx(100.3988)


def test_inside_bar_fake_breakout_reversal_can_long_after_failed_downside_breakout() -> None:
    strategy = InsideBarFakeBreakoutReversalStrategy(
        allow_longs=True,
        allow_shorts=False,
        max_stop_loss_pct=0.006,
    )
    flat_context = StrategyContext(
        in_position=False,
        available_capital=100_000.0,
        is_end_of_day=False,
    )
    rows = [
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-02T09:15:00+05:30"),
                "high": 100.0,
                "low": 95.0,
                "close": 98.0,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-02T09:30:00+05:30"),
                "high": 99.0,
                "low": 96.0,
                "close": 98.5,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-02T09:45:00+05:30"),
                "high": 97.0,
                "low": 94.0,
                "close": 94.5,
            }
        ),
        pd.Series(
            {
                "timestamp": pd.Timestamp("2026-01-02T10:00:00+05:30"),
                "high": 97.0,
                "low": 95.5,
                "close": 96.2,
            }
        ),
    ]

    for row in rows[:-1]:
        assert strategy.generate_signal(row, flat_context) == "HOLD"

    assert strategy.generate_signal(rows[-1], flat_context) == "BUY"
    stop = strategy.entry_stop_loss(rows[-1], "BUY", 1.0)
    assert stop == pytest.approx(95.6228)
