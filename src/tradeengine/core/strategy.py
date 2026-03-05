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
