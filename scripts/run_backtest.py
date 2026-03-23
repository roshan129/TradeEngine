#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from datetime import time

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    FirstFiveMinuteFakeBreakoutStrategy,
    FirstFiveMinuteCandleMomentumStrategy,
    InsideBarBreakoutStrategy,
    MLSignalStrategy,
    OpeningRangeBreakoutStrategy,
    RandomOpenDirectionStrategy,
    SupportResistanceReversalStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
    Strategy,
    VwapRsiMeanReversionStrategy,
)
from tradeengine.ml.models.predictor import ModelPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic candle-by-candle backtest")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing OHLC + indicator columns required by selected strategy",
    )
    parser.add_argument(
        "--structure-input",
        default="",
        help="Optional higher-timeframe CSV used to project structure levels into --input",
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
            "ml_signal",
            "opening_range_breakout",
            "first_five_minute_momentum",
            "first_five_minute_fake_breakout",
            "inside_bar_breakout",
            "random_open_direction",
            "support_resistance_reversal",
            "one_minute_vwap_ema9_scalp",
            "one_minute_vwap_ema9_icici",
        ],
        default="ema_rsi",
        help="Strategy to run (default: ema_rsi)",
    )
    parser.add_argument("--model", default="", help="Optional model artifact for ml_signal")
    parser.add_argument(
        "--predictions-output",
        default="",
        help="Optional CSV path to write predictions with probabilities",
    )
    parser.add_argument(
        "--buy-threshold-proba",
        type=float,
        default=0.0,
        help="Minimum BUY probability for BUY signal (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--sell-threshold-proba",
        type=float,
        default=0.0,
        help="Minimum SELL probability for SELL signal (default: 0.0 = disabled)",
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
        "--stop-after-first-win-per-day",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop taking new entries for the day after the first winning trade (strategy default applies when omitted)",
    )
    parser.add_argument(
        "--opening-start",
        default="09:15",
        help="Opening range start for opening_range_breakout in HH:MM (default: 09:15)",
    )
    parser.add_argument(
        "--opening-end",
        default="09:20",
        help="Opening range end for opening_range_breakout in HH:MM (default: 09:20)",
    )
    parser.add_argument(
        "--orb-sl-pct",
        type=float,
        default=0.0025,
        help="Stop-loss percent for opening_range_breakout (default: 0.0025)",
    )
    parser.add_argument(
        "--orb-tp-pct",
        type=float,
        default=0.0025,
        help="Take-profit percent for opening_range_breakout (default: 0.0025)",
    )
    parser.add_argument(
        "--orb-prob-threshold",
        type=float,
        default=0.65,
        help="Minimum ML probability for opening_range_breakout entries (default: 0.65)",
    )
    parser.add_argument(
        "--orb-use-trend-filter",
        action="store_true",
        help="Enable EMA trend filter for opening_range_breakout entries",
    )
    parser.add_argument(
        "--orb-use-volatility-filter",
        action="store_true",
        help="Enable ATR/BB-width volatility filter for opening_range_breakout entries",
    )
    parser.add_argument(
        "--orb-min-atr-pct",
        type=float,
        default=0.001,
        help="Minimum ATR/close for opening_range_breakout when volatility filter enabled",
    )
    parser.add_argument(
        "--orb-min-bb-width",
        type=float,
        default=0.004,
        help="Minimum BB width for opening_range_breakout when volatility filter enabled",
    )
    parser.add_argument(
        "--first-candle-breakout-start",
        default="09:20",
        help="Breakout monitoring start for first_five_minute_momentum in HH:MM (default: 09:20)",
    )
    parser.add_argument(
        "--first-candle-breakout-end",
        default="10:00",
        help="Breakout monitoring end for first_five_minute_momentum in HH:MM (default: 10:00)",
    )
    parser.add_argument(
        "--first-candle-min-body-percent",
        type=float,
        default=0.5,
        help="Minimum body/range ratio for first_five_minute_momentum (default: 0.5)",
    )
    parser.add_argument(
        "--first-candle-rr-multiple",
        type=float,
        default=1.0,
        help="Risk-reward multiple for first_five_minute_momentum (default: 1.0)",
    )
    parser.add_argument(
        "--first-candle-min-range-atr-multiple",
        type=float,
        default=0.8,
        help="Minimum first-candle range as ATR multiple for first_five_minute_momentum (default: 0.8)",
    )
    parser.add_argument(
        "--first-candle-max-gap-percent",
        type=float,
        default=1.5,
        help="Maximum allowed gap percent for first_five_minute_momentum (default: 1.5)",
    )
    parser.add_argument(
        "--first-candle-disable-volume-filter",
        action="store_true",
        help="Disable breakout volume filter for first_five_minute_momentum",
    )
    parser.add_argument(
        "--first-candle-disable-vwap-filter",
        action="store_true",
        help="Disable VWAP filter for first_five_minute_momentum",
    )
    parser.add_argument(
        "--first-candle-disable-atr-filter",
        action="store_true",
        help="Disable ATR filter for first_five_minute_momentum",
    )
    parser.add_argument(
        "--first-candle-disable-gap-filter",
        action="store_true",
        help="Disable gap filter for first_five_minute_momentum",
    )
    parser.add_argument(
        "--first-candle-stop-loss-pct",
        type=float,
        default=0.0,
        help="Fixed stop-loss percent from entry for first_five_minute_momentum (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--first-candle-take-profit-pct",
        type=float,
        default=0.0,
        help="Fixed take-profit percent from entry for first_five_minute_momentum (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--fake-breakout-failure-deadline",
        default="09:40",
        help="Latest time to observe the initial trap breakout for first_five_minute_fake_breakout (default: 09:40)",
    )
    parser.add_argument(
        "--fake-breakout-trade-deadline",
        default="10:00",
        help="Final time to enter/hold first_five_minute_fake_breakout trades (default: 10:00)",
    )
    parser.add_argument(
        "--fake-breakout-rr-multiple",
        type=float,
        default=1.0,
        help="Risk-reward multiple for first_five_minute_fake_breakout (default: 1.0)",
    )
    parser.add_argument(
        "--fake-breakout-disable-volume-filter",
        action="store_true",
        help="Disable volume filter for first_five_minute_fake_breakout",
    )
    parser.add_argument(
        "--fake-breakout-disable-vwap-filter",
        action="store_true",
        help="Disable VWAP filter for first_five_minute_fake_breakout",
    )
    parser.add_argument(
        "--fake-breakout-stop-loss-pct",
        type=float,
        default=0.0,
        help="Fixed stop-loss percent from entry for first_five_minute_fake_breakout (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--fake-breakout-take-profit-pct",
        type=float,
        default=0.0,
        help="Fixed take-profit percent from entry for first_five_minute_fake_breakout (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--inside-entry-start",
        default="09:15",
        help="Entry window start for inside_bar_breakout in HH:MM (default: 09:15)",
    )
    parser.add_argument(
        "--inside-entry-end",
        default="15:15",
        help="Entry window end for inside_bar_breakout in HH:MM (default: 15:15)",
    )
    parser.add_argument(
        "--inside-max-setup-candles",
        type=int,
        default=5,
        help="Max candles to wait for breakout after inside bar (default: 5)",
    )
    parser.add_argument(
        "--inside-min-range-pct",
        type=float,
        default=0.0035,
        help="Minimum mother bar range as pct of close (default: 0.0035 = 0.35%)",
    )
    parser.add_argument(
        "--inside-use-volume-filter",
        action="store_true",
        help="Require breakout candle volume > rolling volume average",
    )
    parser.add_argument(
        "--inside-use-vwap-filter",
        action="store_true",
        help="Require price above VWAP for longs, below VWAP for shorts",
    )
    parser.add_argument(
        "--inside-use-ema-filter",
        action="store_true",
        help="Require EMA20>EMA50>EMA200 for longs, reversed for shorts",
    )
    parser.add_argument(
        "--inside-use-atr-stop",
        action="store_true",
        help="Use ATR-based stop-loss instead of mother-bar stop",
    )
    parser.add_argument(
        "--inside-atr-stop-multiple",
        type=float,
        default=1.0,
        help="ATR multiple for stop-loss when --inside-use-atr-stop is set",
    )
    parser.add_argument(
        "--inside-rr-multiple",
        type=float,
        default=1.5,
        help="Risk-reward multiple for inside_bar_breakout (default: 1.5)",
    )
    parser.add_argument(
        "--inside-use-inside-range",
        action="store_true",
        help="Use inside-bar high/low for breakout and stop instead of mother bar",
    )
    parser.add_argument(
        "--inside-prob-threshold",
        type=float,
        default=0.0,
        help="Minimum ML probability for inside_bar_breakout entries (default: 0 disables)",
    )
    parser.add_argument(
        "--inside-trade-direction",
        choices=["both", "long", "short"],
        default="both",
        help="Directional entries allowed for inside_bar_breakout (default: both)",
    )
    parser.add_argument(
        "--random-entry-time",
        default="09:15",
        help="Entry candle time for random_open_direction in HH:MM (default: 09:15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Deterministic seed for random_open_direction (default: 42)",
    )
    parser.add_argument(
        "--random-rr-multiple",
        type=float,
        default=1.0,
        help="Risk-reward multiple for random_open_direction (default: 1.0)",
    )
    parser.add_argument(
        "--sr-entry-start",
        default="09:20",
        help="Entry window start for support_resistance_reversal in HH:MM (default: 09:20)",
    )
    parser.add_argument(
        "--sr-entry-end",
        default="14:30",
        help="Entry window end for support_resistance_reversal in HH:MM (default: 14:30)",
    )
    parser.add_argument(
        "--sr-zone-tolerance-pct",
        type=float,
        default=0.0005,
        help="Level clustering tolerance for support_resistance_reversal (default: 0.0005)",
    )
    parser.add_argument(
        "--sr-distance-threshold-pct",
        type=float,
        default=0.0008,
        help="Max distance from level for support_resistance_reversal (default: 0.0008)",
    )
    parser.add_argument(
        "--sr-stop-offset-pct",
        type=float,
        default=0.001,
        help="Stop offset beyond level for support_resistance_reversal (default: 0.001)",
    )
    parser.add_argument(
        "--sr-risk-reward",
        type=float,
        default=1.0,
        help="Risk-reward multiple for support_resistance_reversal (default: 1.0)",
    )
    parser.add_argument(
        "--sr-cooldown-candles",
        type=int,
        default=10,
        help="Cooldown in candles per level for support_resistance_reversal (default: 10)",
    )
    parser.add_argument(
        "--sr-level-expiry-candles",
        type=int,
        default=120,
        help="Expiry in candles for support_resistance_reversal levels (default: 120)",
    )
    parser.add_argument(
        "--sr-min-touch-count",
        type=int,
        default=2,
        help="Minimum touches required for support_resistance_reversal levels (default: 2)",
    )
    parser.add_argument(
        "--sr-volume-multiplier",
        type=float,
        default=1.0,
        help="Volume multiplier vs rolling avg for support_resistance_reversal (default: 1.0)",
    )
    parser.add_argument(
        "--sr-disable-volume-filter",
        action="store_true",
        help="Disable volume filter for support_resistance_reversal",
    )
    parser.add_argument(
        "--sr-disable-vwap-filter",
        action="store_true",
        help="Disable VWAP trend filter for support_resistance_reversal",
    )
    parser.add_argument(
        "--sr-disable-trend-filter",
        action="store_true",
        help="Disable EMA20 slope filter for support_resistance_reversal",
    )
    parser.add_argument(
        "--sr-ema20-slope-threshold",
        type=float,
        default=0.001,
        help="EMA20 slope threshold for support_resistance_reversal (default: 0.001)",
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
    parser.add_argument(
        "--ml-entry-start",
        default="09:20",
        help="Entry window start for ml_signal strategy in HH:MM (default: 09:20)",
    )
    parser.add_argument(
        "--ml-entry-end",
        default="10:20",
        help="Entry window end for ml_signal strategy in HH:MM (default: 10:20)",
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


def _apply_threshold_gating(
    raw_predictions: pd.Series,
    probabilities: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.Series:
    if buy_threshold <= 0.0 and sell_threshold <= 0.0:
        return raw_predictions

    buy_proba = probabilities["buy_probability"]
    sell_proba = probabilities["sell_probability"]

    gated = pd.Series("HOLD", index=raw_predictions.index, name="prediction", dtype="object")
    buy_mask = buy_proba >= buy_threshold
    sell_mask = sell_proba >= sell_threshold

    gated.loc[buy_mask] = "BUY"
    gated.loc[sell_mask] = "SELL"

    conflict_mask = buy_mask & sell_mask
    if conflict_mask.any():
        gated.loc[conflict_mask] = "HOLD"
        gated.loc[conflict_mask & (buy_proba > sell_proba)] = "BUY"
        gated.loc[conflict_mask & (sell_proba > buy_proba)] = "SELL"
    return gated


def _within_zone(price: float, level_price: float, zone_tolerance_pct: float) -> bool:
    if level_price <= 0:
        return False
    return abs(price - level_price) / level_price <= zone_tolerance_pct


def _build_structure_snapshots(
    structure_df: pd.DataFrame,
    zone_tolerance_pct: float,
    level_expiry_candles: int,
    min_touch_count: int,
) -> pd.DataFrame:
    clean = structure_df.copy(deep=True)
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean = clean.sort_values("timestamp", ascending=True).reset_index(drop=True)

    current_day: object | None = None
    bar_index = 0
    levels: list[dict[str, float | int | str]] = []
    buffer: deque[dict[str, float]] = deque(maxlen=5)
    rows: list[dict[str, object]] = []

    def add_level(level_type: str, price: float, index: int) -> None:
        if price <= 0:
            return
        for level in levels:
            if level["level_type"] != level_type:
                continue
            if _within_zone(price, float(level["price"]), zone_tolerance_pct):
                touch_count = int(level["touch_count"]) + 1
                level["price"] = ((float(level["price"]) * int(level["touch_count"])) + price) / touch_count
                level["touch_count"] = touch_count
                level["last_touched_index"] = index
                return
        levels.append(
            {
                "level_type": level_type,
                "price": price,
                "touch_count": 1,
                "last_touched_index": index,
            }
        )

    def pick_level(level_type: str, close_price: float) -> tuple[float | None, int | None]:
        candidates: list[dict[str, float | int | str]] = []
        for level in levels:
            if level["level_type"] != level_type:
                continue
            if int(level["touch_count"]) < min_touch_count:
                continue
            level_price = float(level["price"])
            if level_type == "support" and level_price > close_price:
                continue
            if level_type == "resistance" and level_price < close_price:
                continue
            candidates.append(level)
        if not candidates:
            return None, None
        candidates.sort(
            key=lambda level: (
                int(level["touch_count"]),
                int(level["last_touched_index"]),
                -abs(close_price - float(level["price"])),
            ),
            reverse=True,
        )
        best = candidates[0]
        return float(best["price"]), int(best["touch_count"])

    for _, row in clean.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if pd.isna(timestamp):
            continue
        candle_day = timestamp.date()
        if candle_day != current_day:
            current_day = candle_day
            bar_index = 0
            levels = []
            buffer.clear()

        bar_index += 1
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        buffer.append({"high": high, "low": low})
        if len(buffer) == 5:
            highs = [candle["high"] for candle in buffer]
            lows = [candle["low"] for candle in buffer]
            center_high = highs[2]
            center_low = lows[2]
            center_index = bar_index - 2
            if center_high > highs[0] and center_high > highs[1] and center_high > highs[3] and center_high > highs[4]:
                add_level("resistance", center_high, center_index)
            if center_low < lows[0] and center_low < lows[1] and center_low < lows[3] and center_low < lows[4]:
                add_level("support", center_low, center_index)

        for level in levels:
            touch_price = low if level["level_type"] == "support" else high
            if _within_zone(touch_price, float(level["price"]), zone_tolerance_pct):
                if int(level["last_touched_index"]) != bar_index:
                    level["touch_count"] = int(level["touch_count"]) + 1
                    level["last_touched_index"] = bar_index

        levels = [
            level
            for level in levels
            if (bar_index - int(level["last_touched_index"])) <= max(level_expiry_candles, 1)
        ]

        support_price, support_touch_count = pick_level("support", close)
        resistance_price, resistance_touch_count = pick_level("resistance", close)
        rows.append(
            {
                "timestamp": timestamp,
                "sr_support_price": support_price,
                "sr_support_touch_count": support_touch_count,
                "sr_resistance_price": resistance_price,
                "sr_resistance_touch_count": resistance_touch_count,
            }
        )

    return pd.DataFrame(rows)


def _enrich_execution_with_structure(
    execution_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    zone_tolerance_pct: float,
    level_expiry_candles: int,
    min_touch_count: int,
) -> pd.DataFrame:
    execution = execution_df.copy(deep=True)
    execution["timestamp"] = pd.to_datetime(execution["timestamp"], errors="coerce")
    execution = execution.sort_values("timestamp", ascending=True).reset_index(drop=True)

    snapshots = _build_structure_snapshots(
        structure_df=structure_df,
        zone_tolerance_pct=zone_tolerance_pct,
        level_expiry_candles=level_expiry_candles,
        min_touch_count=min_touch_count,
    )
    snapshots = snapshots.rename(columns={"timestamp": "structure_timestamp"})

    enriched = pd.merge_asof(
        execution,
        snapshots,
        left_on="timestamp",
        right_on="structure_timestamp",
        direction="backward",
    )
    execution_day = enriched["timestamp"].dt.date
    structure_day = pd.to_datetime(enriched["structure_timestamp"], errors="coerce").dt.date
    structure_columns = [
        "sr_support_price",
        "sr_support_touch_count",
        "sr_resistance_price",
        "sr_resistance_touch_count",
    ]
    enriched.loc[execution_day != structure_day, structure_columns] = pd.NA
    enriched = enriched.drop(columns=["structure_timestamp"])
    enriched[structure_columns] = enriched[structure_columns].fillna(0.0)
    return enriched


def main() -> int:
    args = parse_args()
    for threshold_name, threshold_value in (
        ("buy-threshold-proba", args.buy_threshold_proba),
        ("sell-threshold-proba", args.sell_threshold_proba),
        ("orb-prob-threshold", args.orb_prob_threshold),
        ("inside-prob-threshold", args.inside_prob_threshold),
    ):
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(f"{threshold_name} must be in [0, 1], got {threshold_value}")

    df = pd.read_csv(args.input)
    if args.structure_input and args.strategy != "support_resistance_reversal":
        raise ValueError("--structure-input is currently only supported with --strategy support_resistance_reversal")
    if args.model and args.strategy not in {"ml_signal", "opening_range_breakout", "inside_bar_breakout"}:
        raise ValueError("--model is only supported with --strategy ml_signal, opening_range_breakout, or inside_bar_breakout")

    if args.model:
        predictor = ModelPredictor(model_path=args.model)
        raw_predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        predictions = _apply_threshold_gating(
            raw_predictions=raw_predictions,
            probabilities=probabilities,
            buy_threshold=args.buy_threshold_proba,
            sell_threshold=args.sell_threshold_proba,
        )
        df = df.copy(deep=True)
        df["raw_prediction"] = raw_predictions
        df["prediction"] = predictions
        for column in probabilities.columns:
            df[column] = probabilities[column]
        if args.predictions_output:
            df.to_csv(args.predictions_output, index=False)
            print(f"Predictions written to: {args.predictions_output}")

    if args.structure_input:
        structure_df = pd.read_csv(args.structure_input)
        df = _enrich_execution_with_structure(
            execution_df=df,
            structure_df=structure_df,
            zone_tolerance_pct=args.sr_zone_tolerance_pct,
            level_expiry_candles=max(1, args.sr_level_expiry_candles),
            min_touch_count=max(1, args.sr_min_touch_count),
        )

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    from_ts = timestamps.min()
    to_ts = timestamps.max()
    short_enabled = (
        True
        if args.strategy == "opening_range_breakout"
        else args.allow_shorts or args.reverse_signals
    )
    strategy: Strategy
    if args.strategy == "vwap_rsi_reversion":
        strategy = VwapRsiMeanReversionStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "ml_signal":
        strategy = MLSignalStrategy(
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
            entry_session_start=_parse_hhmm(args.ml_entry_start),
            entry_session_end=_parse_hhmm(args.ml_entry_end),
        )
    elif args.strategy == "opening_range_breakout":
        strategy = OpeningRangeBreakoutStrategy(
            opening_start=_parse_hhmm(args.opening_start),
            opening_end=_parse_hhmm(args.opening_end),
            stop_loss_pct=args.orb_sl_pct,
            take_profit_pct=args.orb_tp_pct,
            probability_threshold=args.orb_prob_threshold,
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
            use_trend_filter=args.orb_use_trend_filter,
            use_volatility_filter=args.orb_use_volatility_filter,
            min_atr_pct=args.orb_min_atr_pct,
            min_bb_width=args.orb_min_bb_width,
        )
    elif args.strategy == "first_five_minute_momentum":
        strategy = FirstFiveMinuteCandleMomentumStrategy(
            breakout_start=_parse_hhmm(args.first_candle_breakout_start),
            breakout_end=_parse_hhmm(args.first_candle_breakout_end),
            min_body_percent=args.first_candle_min_body_percent,
            risk_reward_multiple=args.first_candle_rr_multiple,
            fixed_stop_loss_pct=(
                args.first_candle_stop_loss_pct if args.first_candle_stop_loss_pct > 0 else None
            ),
            fixed_take_profit_pct=(
                args.first_candle_take_profit_pct
                if args.first_candle_take_profit_pct > 0
                else None
            ),
            min_range_atr_multiple=args.first_candle_min_range_atr_multiple,
            max_gap_percent=args.first_candle_max_gap_percent,
            use_volume_filter=not args.first_candle_disable_volume_filter,
            use_vwap_filter=not args.first_candle_disable_vwap_filter,
            use_atr_filter=not args.first_candle_disable_atr_filter,
            use_gap_filter=not args.first_candle_disable_gap_filter,
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "first_five_minute_fake_breakout":
        strategy = FirstFiveMinuteFakeBreakoutStrategy(
            failure_deadline=_parse_hhmm(args.fake_breakout_failure_deadline),
            trade_deadline=_parse_hhmm(args.fake_breakout_trade_deadline),
            risk_reward_multiple=args.fake_breakout_rr_multiple,
            fixed_stop_loss_pct=(
                args.fake_breakout_stop_loss_pct if args.fake_breakout_stop_loss_pct > 0 else None
            ),
            fixed_take_profit_pct=(
                args.fake_breakout_take_profit_pct
                if args.fake_breakout_take_profit_pct > 0
                else None
            ),
            use_volume_filter=not args.fake_breakout_disable_volume_filter,
            use_vwap_filter=not args.fake_breakout_disable_vwap_filter,
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "inside_bar_breakout":
        inside_allow_longs = args.inside_trade_direction in {"both", "long"}
        inside_allow_shorts = args.inside_trade_direction in {"both", "short"}
        strategy = InsideBarBreakoutStrategy(
            entry_session_start=_parse_hhmm(args.inside_entry_start),
            entry_session_end=_parse_hhmm(args.inside_entry_end),
            max_setup_candles=max(1, args.inside_max_setup_candles),
            min_mother_range_pct=args.inside_min_range_pct,
            use_volume_filter=args.inside_use_volume_filter,
            use_vwap_trend_filter=args.inside_use_vwap_filter,
            use_ema_trend_filter=args.inside_use_ema_filter,
            use_atr_stop=args.inside_use_atr_stop,
            atr_stop_multiple=args.inside_atr_stop_multiple,
            risk_reward_multiple=args.inside_rr_multiple,
            use_inside_bar_range=args.inside_use_inside_range,
            probability_threshold=args.inside_prob_threshold,
            allow_longs=inside_allow_longs,
            allow_shorts=inside_allow_shorts and short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "random_open_direction":
        strategy = RandomOpenDirectionStrategy(
            entry_time=_parse_hhmm(args.random_entry_time),
            risk_reward_multiple=args.random_rr_multiple,
            seed=args.random_seed,
            allow_shorts=short_enabled,
            reverse_signals=args.reverse_signals,
        )
    elif args.strategy == "support_resistance_reversal":
        strategy = SupportResistanceReversalStrategy(
            entry_session_start=_parse_hhmm(args.sr_entry_start),
            entry_session_end=_parse_hhmm(args.sr_entry_end),
            zone_tolerance_pct=args.sr_zone_tolerance_pct,
            distance_threshold_pct=args.sr_distance_threshold_pct,
            stop_offset_pct=args.sr_stop_offset_pct,
            risk_reward_multiple=args.sr_risk_reward,
            cooldown_candles=max(0, args.sr_cooldown_candles),
            level_expiry_candles=max(1, args.sr_level_expiry_candles),
            min_touch_count=max(1, args.sr_min_touch_count),
            volume_multiplier=args.sr_volume_multiplier,
            use_volume_filter=not args.sr_disable_volume_filter,
            use_vwap_filter=not args.sr_disable_vwap_filter,
            use_trend_filter=not args.sr_disable_trend_filter,
            ema20_slope_threshold_pct=args.sr_ema20_slope_threshold,
            use_external_levels=bool(args.structure_input),
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

    max_entries = args.max_entries_per_day if args.max_entries_per_day > 0 else None
    if args.strategy == "opening_range_breakout" and max_entries is None:
        max_entries = 3
    if args.strategy == "first_five_minute_momentum" and max_entries is None:
        max_entries = 1
    if args.strategy == "first_five_minute_fake_breakout" and max_entries is None:
        max_entries = 1
    if args.strategy == "support_resistance_reversal" and max_entries is None:
        max_entries = 10
    if args.strategy == "inside_bar_breakout" and max_entries is None:
        max_entries = 2

    stop_after_first_win = args.stop_after_first_win_per_day
    if stop_after_first_win is None:
        stop_after_first_win = args.strategy == "inside_bar_breakout"
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_atr_multiple=args.stop_atr_multiple,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
        allow_shorts=short_enabled,
        max_entries_per_day=max_entries,
        stop_after_first_win_per_day=stop_after_first_win,
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
