from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

Signal = Literal["BUY", "SELL", "SHORT", "COVER", "HOLD"]
PositionSide = Literal["LONG", "SHORT"]


def reverse_signal(signal: Signal) -> Signal:
    mapping: dict[Signal, Signal] = {
        "BUY": "SHORT",
        "SHORT": "BUY",
        "SELL": "COVER",
        "COVER": "SELL",
        "HOLD": "HOLD",
    }
    return mapping[signal]


@dataclass(frozen=True)
class StrategyContext:
    """Per-candle execution context passed into stateless strategies."""

    in_position: bool
    available_capital: float
    is_end_of_day: bool
    position_side: PositionSide | None = None
    position_entry_price: float | None = None
    position_stop_loss: float | None = None


class Strategy(Protocol):
    """Strategy contract for deterministic, row-wise signal generation."""

    required_columns: tuple[str, ...]

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        """Return one of BUY, SELL, or HOLD for the current candle."""

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        """Return stop-loss price for a new BUY/SHORT entry, or None to skip entry."""


@dataclass(frozen=True)
class BaselineEmaRsiStrategy:
    """Baseline long-only strategy used to validate backtesting plumbing."""

    required_columns: tuple[str, ...] = ("ema20", "ema50", "rsi", "atr")
    entry_rsi_threshold: float = 55.0
    exit_rsi_threshold: float = 45.0
    allow_shorts: bool = False
    reverse_signals: bool = False

    @staticmethod
    def _reverse_signal(signal: Signal) -> Signal:
        return reverse_signal(signal)

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ema20 = float(row.get("ema20", float("nan")))
        ema50 = float(row.get("ema50", float("nan")))
        rsi = float(row.get("rsi", float("nan")))

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(rsi):
            return "HOLD"

        signal: Signal = "HOLD"
        if context.in_position:
            if context.position_side == "SHORT":
                if rsi > self.entry_rsi_threshold or context.is_end_of_day:
                    signal = "COVER"
                else:
                    signal = "HOLD"
            elif rsi < self.exit_rsi_threshold or context.is_end_of_day:
                signal = "SELL"
            else:
                signal = "HOLD"
        elif ema20 > ema50 and rsi > self.entry_rsi_threshold:
            signal = "BUY"
        elif self.allow_shorts and ema20 < ema50 and rsi < self.exit_rsi_threshold:
            signal = "SHORT"

        if self.reverse_signals:
            return self._reverse_signal(signal)
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        close = float(row.get("close", float("nan")))
        atr = float(row.get("atr", float("nan")))
        if pd.isna(close) or pd.isna(atr):
            return None
        if atr <= 0:
            return None

        if signal == "BUY":
            return close - (atr * stop_atr_multiple)
        if signal == "SHORT":
            return close + (atr * stop_atr_multiple)
        return None


@dataclass(frozen=True)
class VwapRsiMeanReversionStrategy:
    """Long-only VWAP reversion strategy."""

    required_columns: tuple[str, ...] = ("vwap", "rsi")
    entry_rsi_threshold: float = 35.0
    short_entry_rsi_threshold: float = 65.0
    allow_shorts: bool = False
    reverse_signals: bool = False

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        rsi = float(row.get("rsi", float("nan")))
        if pd.isna(close) or pd.isna(vwap) or pd.isna(rsi):
            return "HOLD"

        signal: Signal = "HOLD"
        if context.in_position:
            if context.position_side == "SHORT":
                signal = "COVER" if (close <= vwap or context.is_end_of_day) else "HOLD"
            else:
                signal = "SELL" if (close >= vwap or context.is_end_of_day) else "HOLD"
        elif close < vwap and rsi < self.entry_rsi_threshold:
            signal = "BUY"
        elif self.allow_shorts and close > vwap and rsi > self.short_entry_rsi_threshold:
            signal = "SHORT"

        if self.reverse_signals:
            return reverse_signal(signal)
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        if signal not in {"BUY", "SHORT"}:
            return None

        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        if pd.isna(close) or pd.isna(vwap):
            return None
        distance = abs(vwap - close)
        if distance <= 0:
            return None

        # 1:1 RR with TP at VWAP distance: stop is mirrored by side.
        if signal == "BUY":
            return close - distance
        return close + distance


@dataclass(frozen=True)
class OneMinuteVwapEma9ScalpStrategy:
    """1-minute VWAP+EMA9 scalping strategy with volume confirmation."""

    required_columns: tuple[str, ...] = ("vwap", "ema9", "rolling_volume_avg", "volume", "atr")
    allow_shorts: bool = True
    reverse_signals: bool = False
    volume_spike_multiplier: float = 1.5
    stop_min_pct: float = 0.002
    stop_max_pct: float = 0.003
    risk_reward_multiple: float = 1.2
    take_profit_mode: Literal["rr", "atr"] = "rr"

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        close = float(row.get("close", float("nan")))
        open_price = float(row.get("open", float("nan")))
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        ema9 = float(row.get("ema9", float("nan")))
        atr = float(row.get("atr", float("nan")))
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        inputs = (close, open_price, high, low, vwap, ema9, atr, volume, volume_avg)

        if any(pd.isna(value) for value in inputs):
            return "HOLD"
        if volume_avg <= 0 or atr <= 0:
            return "HOLD"

        signal: Signal = "HOLD"
        if context.in_position:
            signal = self._exit_signal(
                close=close,
                atr=atr,
                context=context,
            )
        else:
            long_setup = (
                close > vwap
                and low <= ema9
                and close > ema9
                and close > open_price
                and volume > (self.volume_spike_multiplier * volume_avg)
            )
            short_setup = (
                close < vwap
                and high >= ema9
                and close < ema9
                and close < open_price
                and volume > (self.volume_spike_multiplier * volume_avg)
            )
            if long_setup:
                signal = "BUY"
            elif self.allow_shorts and short_setup:
                signal = "SHORT"

        if self.reverse_signals:
            return reverse_signal(signal)
        return signal

    def _exit_signal(self, close: float, atr: float, context: StrategyContext) -> Signal:
        if context.position_side is None:
            return "HOLD"
        if context.is_end_of_day:
            return "SELL" if context.position_side == "LONG" else "COVER"
        if context.position_entry_price is None or context.position_stop_loss is None:
            return "HOLD"

        entry_price = context.position_entry_price
        stop_loss = context.position_stop_loss
        if context.position_side == "LONG":
            risk = entry_price - stop_loss
            if risk <= 0:
                return "HOLD"
            if self.take_profit_mode == "atr":
                target = entry_price + atr
            else:
                target = entry_price + (self.risk_reward_multiple * risk)
            return "SELL" if close >= target else "HOLD"

        risk = stop_loss - entry_price
        if risk <= 0:
            return "HOLD"
        if self.take_profit_mode == "atr":
            target = entry_price - atr
        else:
            target = entry_price - (self.risk_reward_multiple * risk)
        return "COVER" if close <= target else "HOLD"

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        close = float(row.get("close", float("nan")))
        low = float(row.get("low", float("nan")))
        high = float(row.get("high", float("nan")))
        if any(pd.isna(value) for value in (close, low, high)):
            return None
        if close <= 0:
            return None

        min_distance = close * self.stop_min_pct
        max_distance = close * self.stop_max_pct
        if min_distance <= 0 or max_distance <= 0:
            return None

        if signal == "BUY":
            swing_distance = max(close - low, 0.0)
            distance = min(max(swing_distance, min_distance), max_distance)
            return close - distance
        if signal == "SHORT":
            swing_distance = max(high - close, 0.0)
            distance = min(max(swing_distance, min_distance), max_distance)
            return close + distance
        return None
