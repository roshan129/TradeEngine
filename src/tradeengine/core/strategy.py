from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import time
import random
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
class MLSignalStrategy:
    """Strategy that executes model-provided BUY/SELL/HOLD predictions."""

    required_columns: tuple[str, ...] = ("prediction", "atr")
    non_numeric_columns: tuple[str, ...] = ("prediction",)
    allow_shorts: bool = False
    reverse_signals: bool = False
    entry_session_start: time = time(hour=9, minute=20)
    entry_session_end: time = time(hour=10, minute=20)

    def _in_entry_window(self, row: pd.Series) -> bool:
        ts = row.get("timestamp")
        if ts is None:
            return True
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return True
        candle_time = timestamp.time()
        return self.entry_session_start <= candle_time <= self.entry_session_end

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        raw_prediction = str(row.get("prediction", "HOLD")).upper()
        if raw_prediction not in {"BUY", "SELL", "HOLD"}:
            signal: Signal = "HOLD"
        elif raw_prediction == "BUY":
            signal = "BUY"
        elif raw_prediction == "SELL":
            signal = "SELL"
        else:
            signal = "HOLD"

        if context.in_position:
            if context.position_side == "LONG":
                if signal == "SELL" or context.is_end_of_day:
                    signal = "SELL"
                else:
                    signal = "HOLD"
            elif context.position_side == "SHORT":
                if signal == "BUY" or context.is_end_of_day:
                    signal = "COVER"
                else:
                    signal = "HOLD"
        else:
            if not self._in_entry_window(row):
                signal = "HOLD"
            if signal == "SELL":
                signal = "SHORT" if self.allow_shorts else "HOLD"

        if self.reverse_signals:
            return reverse_signal(signal)
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
        if close <= 0 or atr <= 0:
            return None

        if signal == "BUY":
            return close - (atr * stop_atr_multiple)
        if signal == "SHORT":
            return close + (atr * stop_atr_multiple)
        return None


@dataclass
class OpeningRangeBreakoutStrategy:
    """Opening range breakout with optional ML probability filter."""

    opening_start: time = time(hour=9, minute=15)
    opening_end: time = time(hour=9, minute=20)
    stop_loss_pct: float = 0.0025
    take_profit_pct: float = 0.0025
    probability_threshold: float = 0.65
    allow_shorts: bool = True
    reverse_signals: bool = False
    use_trend_filter: bool = False
    use_volatility_filter: bool = False
    min_atr_pct: float = 0.001
    min_bb_width: float = 0.004
    buy_probability_column: str = "buy_probability"
    sell_probability_column: str = "sell_probability"
    _current_day: object | None = field(default=None, init=False, repr=False)
    _opening_high: float | None = field(default=None, init=False, repr=False)
    _opening_low: float | None = field(default=None, init=False, repr=False)
    _opening_complete: bool = field(default=False, init=False, repr=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        if self.probability_threshold and self.probability_threshold > 0:
            columns.extend([self.buy_probability_column, self.sell_probability_column])
        if self.use_volatility_filter:
            columns.extend(["atr", "bb_width", "close"])
        if self.use_trend_filter:
            columns.extend(["ema20", "ema50", "ema200"])
        return tuple(dict.fromkeys(columns))

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._opening_high = None
        self._opening_low = None
        self._opening_complete = False

    def _update_opening_range(self, row: pd.Series, candle_time: time) -> None:
        if self.opening_start <= candle_time <= self.opening_end:
            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return
            if self._opening_high is None or high > self._opening_high:
                self._opening_high = high
            if self._opening_low is None or low < self._opening_low:
                self._opening_low = low

        if candle_time > self.opening_end:
            self._opening_complete = True

    def _probability_ok(self, row: pd.Series, side: Signal) -> bool:
        if not self.probability_threshold or self.probability_threshold <= 0:
            return True
        column = (
            self.buy_probability_column if side == "BUY" else self.sell_probability_column
        )
        proba = float(row.get(column, float("nan")))
        if pd.isna(proba):
            return False
        return proba >= self.probability_threshold

    def _resolve_conflict(self, row: pd.Series) -> Signal:
        if not self.probability_threshold or self.probability_threshold <= 0:
            return "HOLD"
        buy_proba = float(row.get(self.buy_probability_column, float("nan")))
        sell_proba = float(row.get(self.sell_probability_column, float("nan")))
        if pd.isna(buy_proba) or pd.isna(sell_proba):
            return "HOLD"
        if buy_proba > sell_proba:
            return "BUY"
        if sell_proba > buy_proba:
            return "SHORT"
        return "HOLD"

    def _volatility_ok(self, row: pd.Series) -> bool:
        if not self.use_volatility_filter:
            return True
        close = float(row.get("close", float("nan")))
        atr = float(row.get("atr", float("nan")))
        bb_width = float(row.get("bb_width", float("nan")))
        if pd.isna(close) or pd.isna(atr) or pd.isna(bb_width) or close <= 0:
            return False
        atr_pct = atr / close
        return atr_pct >= self.min_atr_pct and bb_width >= self.min_bb_width

    def _trend_ok(self, row: pd.Series, side: Signal) -> bool:
        if not self.use_trend_filter:
            return True
        ema20 = float(row.get("ema20", float("nan")))
        ema50 = float(row.get("ema50", float("nan")))
        ema200 = float(row.get("ema200", float("nan")))
        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ema200):
            return False
        if side == "BUY":
            return ema20 > ema50 > ema200
        if side == "SELL":
            return ema20 < ema50 < ema200
        return False

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        self._update_opening_range(row, candle_time)

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if context.is_end_of_day:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None:
                return "HOLD"

            entry_price = context.position_entry_price
            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"

            if context.position_side == "LONG":
                target = entry_price * (1.0 + self.take_profit_pct)
                return "SELL" if high >= target else "HOLD"

            target = entry_price * (1.0 - self.take_profit_pct)
            return "COVER" if low <= target else "HOLD"

        if not self._opening_complete:
            return "HOLD"
        if self._opening_high is None or self._opening_low is None:
            return "HOLD"

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        if pd.isna(high) or pd.isna(low):
            return "HOLD"

        if not self._volatility_ok(row):
            return "HOLD"

        long_trigger = high >= self._opening_high
        short_trigger = low <= self._opening_low

        signal: Signal = "HOLD"
        if long_trigger and self._probability_ok(row, "BUY") and self._trend_ok(row, "BUY"):
            signal = "BUY"
        if (
            short_trigger
            and self.allow_shorts
            and self._probability_ok(row, "SELL")
            and self._trend_ok(row, "SELL")
        ):
            signal = "SHORT" if signal == "HOLD" else self._resolve_conflict(row)
        if long_trigger and short_trigger and signal == "HOLD":
            signal = self._resolve_conflict(row)

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
        close = float(row.get("close", float("nan")))
        if pd.isna(close) or close <= 0:
            return None

        if signal == "BUY":
            return close * (1.0 - self.stop_loss_pct)
        if signal == "SHORT":
            return close * (1.0 + self.stop_loss_pct)
        return None


@dataclass
class OpeningRangePullbackStrategy:
    """Trade a pullback to the opening-range breakout level after confirmation."""

    opening_start: time = time(hour=9, minute=15)
    opening_end: time = time(hour=9, minute=25)
    entry_end: time = time(hour=11, minute=0)
    session_exit_time: time = time(hour=15, minute=15)
    risk_reward_multiple: float = 1.5
    pullback_tolerance_pct: float = 0.001
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _opening_high: float | None = field(default=None, init=False, repr=False)
    _opening_low: float | None = field(default=None, init=False, repr=False)
    _opening_complete: bool = field(default=False, init=False, repr=False)
    _breakout_side: Signal = field(default="HOLD", init=False, repr=False)
    _breakout_level: float | None = field(default=None, init=False, repr=False)
    _breakout_timestamp: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _trade_taken: bool = field(default=False, init=False, repr=False)
    _pending_stop_loss: float | None = field(default=None, init=False, repr=False)

    required_columns: tuple[str, ...] = ()

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._opening_high = None
        self._opening_low = None
        self._opening_complete = False
        self._breakout_side = "HOLD"
        self._breakout_level = None
        self._breakout_timestamp = None
        self._trade_taken = False
        self._pending_stop_loss = None

    def _update_opening_range(self, row: pd.Series, candle_time: time) -> None:
        if self.opening_start <= candle_time <= self.opening_end:
            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return
            if self._opening_high is None or high > self._opening_high:
                self._opening_high = high
            if self._opening_low is None or low < self._opening_low:
                self._opening_low = low

        if candle_time > self.opening_end:
            self._opening_complete = True

    def _set_breakout(self, side: Signal, timestamp: pd.Timestamp) -> None:
        self._breakout_side = side
        self._breakout_timestamp = timestamp
        self._breakout_level = self._opening_high if side == "BUY" else self._opening_low

    def _clear_breakout(self) -> None:
        self._breakout_side = "HOLD"
        self._breakout_level = None
        self._breakout_timestamp = None

    def _pullback_signal(self, row: pd.Series) -> Signal:
        if (
            self._breakout_side not in {"BUY", "SHORT"}
            or self._breakout_level is None
            or self._opening_high is None
            or self._opening_low is None
        ):
            return "HOLD"

        close = float(row.get("close", float("nan")))
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        if pd.isna(close) or pd.isna(high) or pd.isna(low):
            return "HOLD"

        tolerance = abs(self._breakout_level) * max(self.pullback_tolerance_pct, 0.0)

        if self._breakout_side == "BUY":
            if close < self._opening_low:
                self._clear_breakout()
                return "HOLD"
            if low <= self._breakout_level + tolerance and close >= self._breakout_level:
                self._pending_stop_loss = self._opening_low
                return "BUY"
            return "HOLD"

        if close > self._opening_high:
            self._clear_breakout()
            return "HOLD"
        if high >= self._breakout_level - tolerance and close <= self._breakout_level:
            self._pending_stop_loss = self._opening_high
            return "SHORT"
        return "HOLD"

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        self._update_opening_range(row, candle_time)

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if context.is_end_of_day or candle_time >= self.session_exit_time:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"

            if context.position_side == "LONG":
                target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"

            target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        if self._trade_taken or not self._opening_complete:
            return "HOLD"
        if self._opening_high is None or self._opening_low is None:
            return "HOLD"
        if candle_time > self.entry_end:
            return "HOLD"

        if self._breakout_side == "HOLD":
            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"
            long_breakout = high > self._opening_high
            short_breakout = low < self._opening_low
            if long_breakout and short_breakout:
                return "HOLD"
            if long_breakout:
                self._set_breakout("BUY", timestamp)
                return "HOLD"
            if short_breakout and self.allow_shorts:
                self._set_breakout("SHORT", timestamp)
                return "HOLD"
            return "HOLD"

        if self._breakout_timestamp is not None and timestamp <= self._breakout_timestamp:
            return "HOLD"

        signal = self._pullback_signal(row)
        if signal == "HOLD":
            return "HOLD"

        self._trade_taken = True
        if self.reverse_signals:
            reversed_signal = reverse_signal(signal)
            self._pending_stop_loss = (
                self._opening_low if reversed_signal == "BUY" else self._opening_high
            )
            return reversed_signal
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del row, signal, stop_atr_multiple
        stop_loss = self._pending_stop_loss
        self._pending_stop_loss = None
        return stop_loss


@dataclass
class VwapTrendContinuationStrategy:
    """Trade shallow continuation pullbacks that hold above a rising VWAP."""

    entry_session_start: time = time(hour=9, minute=20)
    entry_session_end: time = time(hour=11, minute=0)
    session_exit_time: time = time(hour=15, minute=15)
    exit_mode: Literal["rr", "vwap_break", "trailing_low"] = "vwap_break"
    risk_reward_multiple: float = 2.0
    min_vwap_slope_pct: float = 0.0001
    vwap_slope_lookback_bars: int = 3
    min_candles_above_vwap: int = 3
    min_distance_above_vwap_pct: float = 0.001
    pullback_lookback_bars: int = 5
    min_pullback_size_pct: float = 0.001
    max_pullback_size_pct: float = 0.002
    fixed_stop_loss_pct: float = 0.003
    pullback_low_buffer_pct: float = 0.0
    min_atr_pct: float = 0.0
    min_bb_width: float = 0.0
    use_ema_trend_filter: bool = False
    allow_shorts: bool = False
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _previous_close: float | None = field(default=None, init=False, repr=False)
    _previous_vwap: float | None = field(default=None, init=False, repr=False)
    _previous_high: float | None = field(default=None, init=False, repr=False)
    _previous_low: float | None = field(default=None, init=False, repr=False)
    _recent_closes: deque[float] = field(default_factory=deque, init=False, repr=False)
    _recent_vwaps: deque[float] = field(default_factory=deque, init=False, repr=False)
    _recent_highs: deque[float] = field(default_factory=deque, init=False, repr=False)
    _recent_lows: deque[float] = field(default_factory=deque, init=False, repr=False)
    _pending_setup_high: float | None = field(default=None, init=False, repr=False)
    _pending_setup_low: float | None = field(default=None, init=False, repr=False)
    _pending_stop_loss: float | None = field(default=None, init=False, repr=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = ["open", "high", "low", "close", "vwap"]
        if self.use_ema_trend_filter:
            columns.extend(["ema20", "ema50"])
        if self.min_atr_pct > 0:
            columns.append("atr")
        if self.min_bb_width > 0:
            columns.append("bb_width")
        return tuple(columns)

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._previous_close = None
        self._previous_vwap = None
        self._previous_high = None
        self._previous_low = None
        self._recent_closes.clear()
        self._recent_vwaps.clear()
        self._recent_highs.clear()
        self._recent_lows.clear()
        self._pending_setup_high = None
        self._pending_setup_low = None
        self._pending_stop_loss = None

    def _finalize_row(self, close: float, vwap: float, high: float, low: float) -> None:
        self._previous_close = close
        self._previous_vwap = vwap
        self._previous_high = high
        self._previous_low = low
        self._recent_closes.append(close)
        self._recent_vwaps.append(vwap)
        self._recent_highs.append(high)
        self._recent_lows.append(low)
        max_history = max(
            2,
            self.vwap_slope_lookback_bars + 1,
            self.min_candles_above_vwap,
            self.pullback_lookback_bars,
        )
        while len(self._recent_closes) > max_history:
            self._recent_closes.popleft()
        while len(self._recent_vwaps) > max_history:
            self._recent_vwaps.popleft()
        while len(self._recent_highs) > max_history:
            self._recent_highs.popleft()
        while len(self._recent_lows) > max_history:
            self._recent_lows.popleft()

    def _in_entry_session(self, candle_time: time) -> bool:
        return self.entry_session_start <= candle_time <= self.entry_session_end

    def _vwap_slope_ok(self, vwap: float, side: Signal) -> bool:
        lookback = max(1, self.vwap_slope_lookback_bars)
        if len(self._recent_vwaps) < lookback:
            return False
        reference_vwap = list(self._recent_vwaps)[-lookback]
        if reference_vwap <= 0:
            return False
        slope_pct = (vwap - reference_vwap) / reference_vwap
        if side == "BUY":
            return slope_pct >= self.min_vwap_slope_pct
        if side == "SHORT":
            return slope_pct <= -self.min_vwap_slope_pct
        return False

    def _recent_strength_ok(self) -> bool:
        lookback = max(1, self.min_candles_above_vwap)
        if len(self._recent_lows) < lookback or len(self._recent_vwaps) < lookback:
            return False

        recent_lows = list(self._recent_lows)[-lookback:]
        recent_closes = list(self._recent_closes)[-lookback:]
        recent_vwaps = list(self._recent_vwaps)[-lookback:]
        for low, close, vwap in zip(recent_lows, recent_closes, recent_vwaps, strict=False):
            if vwap <= 0:
                return False
            min_price = vwap * (1.0 + self.min_distance_above_vwap_pct)
            if low <= vwap or close <= min_price:
                return False
        return True

    def _regime_ok(self, row: pd.Series, close: float) -> bool:
        atr = float(row.get("atr", float("nan")))
        bb_width = float(row.get("bb_width", float("nan")))
        if self.min_atr_pct > 0:
            if pd.isna(atr) or close <= 0 or (atr / close) < self.min_atr_pct:
                return False
        if self.min_bb_width > 0:
            if pd.isna(bb_width) or bb_width < self.min_bb_width:
                return False
        return True

    def _trend_filter_ok(self, row: pd.Series) -> bool:
        if not self.use_ema_trend_filter:
            return True
        ema20 = float(row.get("ema20", float("nan")))
        ema50 = float(row.get("ema50", float("nan")))
        if pd.isna(ema20) or pd.isna(ema50):
            return False
        return ema20 > ema50

    def _pullback_size_pct(self, low: float) -> float | None:
        lookback = max(1, self.pullback_lookback_bars)
        if len(self._recent_highs) < lookback:
            return None
        swing_high = max(list(self._recent_highs)[-lookback:])
        if swing_high <= 0 or low >= swing_high:
            return None
        return (swing_high - low) / swing_high

    def _prepare_long_setup(self, row: pd.Series) -> bool:
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        if any(pd.isna(value) for value in (high, low, close, vwap)):
            return False
        if close <= 0 or vwap <= 0:
            return False
        if close <= vwap or low <= vwap:
            return False
        if not self._trend_filter_ok(row):
            return False
        if not self._regime_ok(row, close):
            return False
        if not self._recent_strength_ok():
            return False
        if not self._vwap_slope_ok(vwap, "BUY"):
            return False
        if self._previous_close is not None and close >= self._previous_close:
            return False

        close_distance_pct = (close - vwap) / vwap
        if close_distance_pct < self.min_distance_above_vwap_pct:
            return False

        pullback_size_pct = self._pullback_size_pct(low)
        if pullback_size_pct is None:
            return False
        if pullback_size_pct < self.min_pullback_size_pct:
            return False
        if pullback_size_pct > self.max_pullback_size_pct:
            return False

        self._pending_setup_high = high
        self._pending_setup_low = low
        return True

    @staticmethod
    def _unrealized_pnl_pct(
        close: float,
        entry_price: float,
        side: PositionSide,
    ) -> float:
        if entry_price <= 0:
            return 0.0
        if side == "LONG":
            return (close - entry_price) / entry_price
        return (entry_price - close) / entry_price

    def _compute_entry_stop_loss(self, entry_close: float) -> float | None:
        if self._pending_setup_low is None:
            return None

        stop_loss = self._pending_setup_low * (1.0 - self.pullback_low_buffer_pct)
        if self.fixed_stop_loss_pct > 0 and entry_close > 0:
            fixed_stop = entry_close * (1.0 - self.fixed_stop_loss_pct)
            stop_loss = max(stop_loss, fixed_stop)
        return stop_loss

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        if pd.isna(high) or pd.isna(low) or pd.isna(close) or pd.isna(vwap):
            return "HOLD"

        signal: Signal = "HOLD"
        try:
            if context.in_position:
                if context.position_side is None:
                    return "HOLD"
                if context.is_end_of_day or candle_time >= self.session_exit_time:
                    signal = "SELL" if context.position_side == "LONG" else "COVER"
                    return reverse_signal(signal) if self.reverse_signals else signal
                if context.position_entry_price is None or context.position_stop_loss is None:
                    return "HOLD"

                entry_price = context.position_entry_price
                stop_loss = context.position_stop_loss
                risk = abs(entry_price - stop_loss)
                if risk <= 0:
                    return "HOLD"

                if self.exit_mode == "rr":
                    if context.position_side == "LONG":
                        target = entry_price + (risk * self.risk_reward_multiple)
                        if high >= target:
                            signal = "SELL"
                            return reverse_signal(signal) if self.reverse_signals else signal
                    else:
                        target = entry_price - (risk * self.risk_reward_multiple)
                        if low <= target:
                            signal = "COVER"
                            return reverse_signal(signal) if self.reverse_signals else signal
                elif self.exit_mode == "vwap_break":
                    if context.position_side == "LONG" and close < vwap:
                        signal = "SELL"
                        return reverse_signal(signal) if self.reverse_signals else signal
                    if context.position_side == "SHORT" and close > vwap:
                        signal = "COVER"
                        return reverse_signal(signal) if self.reverse_signals else signal
                elif self.exit_mode == "trailing_low":
                    if context.position_side == "LONG" and self._previous_low is not None:
                        if close < self._previous_low:
                            signal = "SELL"
                            return reverse_signal(signal) if self.reverse_signals else signal
                    if context.position_side == "SHORT" and self._previous_high is not None:
                        if close > self._previous_high:
                            signal = "COVER"
                            return reverse_signal(signal) if self.reverse_signals else signal
                return reverse_signal(signal) if self.reverse_signals else signal

            if not self._in_entry_session(candle_time):
                return "HOLD"
            if self._previous_vwap is None:
                return "HOLD"

            if (
                self._pending_setup_high is not None
                and high > self._pending_setup_high
                and close > vwap
                and low > vwap
                and self._trend_filter_ok(row)
                and self._regime_ok(row, close)
                and self._vwap_slope_ok(vwap, "BUY")
            ):
                self._pending_stop_loss = self._compute_entry_stop_loss(close)
                signal = "BUY" if self._pending_stop_loss is not None else "HOLD"
                self._pending_setup_high = None
                self._pending_setup_low = None
                return reverse_signal(signal) if self.reverse_signals else signal

            self._pending_setup_high = None
            self._pending_setup_low = None
            self._prepare_long_setup(row)
            return reverse_signal(signal) if self.reverse_signals else signal
        finally:
            self._finalize_row(close, vwap, high, low)

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del row, signal, stop_atr_multiple
        stop_loss = self._pending_stop_loss
        self._pending_stop_loss = None
        return stop_loss


@dataclass
class FirstFiveMinuteCandleMomentumStrategy:
    """Trade the breakout of the first five-minute candle using one-minute execution."""

    opening_start: time = time(hour=9, minute=15)
    breakout_start: time = time(hour=9, minute=20)
    breakout_end: time = time(hour=10, minute=0)
    min_body_percent: float = 0.5
    risk_reward_multiple: float = 1.0
    fixed_stop_loss_pct: float | None = None
    fixed_take_profit_pct: float | None = None
    min_range_atr_multiple: float = 0.8
    max_gap_percent: float = 1.5
    use_volume_filter: bool = True
    use_vwap_filter: bool = True
    use_atr_filter: bool = True
    use_gap_filter: bool = True
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _opening_open: float | None = field(default=None, init=False, repr=False)
    _opening_high: float | None = field(default=None, init=False, repr=False)
    _opening_low: float | None = field(default=None, init=False, repr=False)
    _opening_close: float | None = field(default=None, init=False, repr=False)
    _opening_ready: bool = field(default=False, init=False, repr=False)
    _opening_direction: Signal = field(default="HOLD", init=False, repr=False)
    _setup_valid: bool = field(default=False, init=False, repr=False)
    _trade_taken: bool = field(default=False, init=False, repr=False)
    _pending_stop_loss: float | None = field(default=None, init=False, repr=False)

    required_columns: tuple[str, ...] = (
        "volume",
        "rolling_volume_avg",
        "vwap",
        "atr",
        "gap_percent",
    )

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._opening_open = None
        self._opening_high = None
        self._opening_low = None
        self._opening_close = None
        self._opening_ready = False
        self._opening_direction = "HOLD"
        self._setup_valid = False
        self._trade_taken = False
        self._pending_stop_loss = None

    def _update_opening_candle(self, row: pd.Series, candle_time: time) -> None:
        if not (self.opening_start <= candle_time < self.breakout_start):
            if candle_time >= self.breakout_start and self._opening_high is not None:
                self._opening_ready = True
            return

        open_price = float(row.get("open", float("nan")))
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        if any(pd.isna(value) for value in (open_price, high, low, close)):
            return

        if self._opening_open is None:
            self._opening_open = open_price
        self._opening_high = high if self._opening_high is None else max(self._opening_high, high)
        self._opening_low = low if self._opening_low is None else min(self._opening_low, low)
        self._opening_close = close

    def _prepare_setup(self, row: pd.Series) -> None:
        if self._opening_open is None or self._opening_high is None:
            return
        if self._opening_low is None or self._opening_close is None:
            return

        opening_range = self._opening_high - self._opening_low
        if opening_range <= 0:
            return

        body = self._opening_close - self._opening_open
        body_percent = abs(body) / opening_range
        direction: Signal = "HOLD"
        if body > 0:
            direction = "BUY"
        elif body < 0:
            direction = "SHORT"

        if direction == "HOLD" or body_percent <= self.min_body_percent:
            return

        atr = float(row.get("atr", float("nan")))
        gap_percent = abs(float(row.get("gap_percent", float("nan"))))
        if self.use_atr_filter and (pd.isna(atr) or atr <= 0 or opening_range < (atr * self.min_range_atr_multiple)):
            return
        if self.use_gap_filter and (pd.isna(gap_percent) or gap_percent > self.max_gap_percent):
            return

        self._opening_direction = direction
        self._setup_valid = True

    def _entry_filters_ok(self, row: pd.Series, signal: Signal) -> bool:
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        close = float(row.get("close", float("nan")))

        if self.use_volume_filter:
            if pd.isna(volume) or pd.isna(volume_avg) or volume_avg <= 0 or volume <= volume_avg:
                return False

        if self.use_vwap_filter:
            if pd.isna(vwap) or pd.isna(close):
                return False
            if signal == "BUY" and close <= vwap:
                return False
            if signal == "SHORT" and close >= vwap:
                return False

        return True

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        self._update_opening_candle(row, candle_time)
        if self._opening_ready and not self._setup_valid and not self._trade_taken:
            self._prepare_setup(row)

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if context.is_end_of_day or candle_time > self.breakout_end:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"

            if context.position_side == "LONG":
                if self.fixed_take_profit_pct is not None and self.fixed_take_profit_pct > 0:
                    target = entry_price * (1.0 + self.fixed_take_profit_pct)
                else:
                    target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"

            if self.fixed_take_profit_pct is not None and self.fixed_take_profit_pct > 0:
                target = entry_price * (1.0 - self.fixed_take_profit_pct)
            else:
                target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        if (
            not self._setup_valid
            or self._trade_taken
            or candle_time < self.breakout_start
            or candle_time > self.breakout_end
        ):
            return "HOLD"

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        if pd.isna(high) or pd.isna(low):
            return "HOLD"

        signal: Signal = "HOLD"
        if self._opening_direction == "BUY" and high >= float(self._opening_high):
            signal = "BUY"
        elif self._opening_direction == "SHORT" and self.allow_shorts and low <= float(self._opening_low):
            signal = "SHORT"

        if signal == "HOLD" or not self._entry_filters_ok(row, signal):
            return "HOLD"

        if self.reverse_signals:
            signal = reverse_signal(signal)

        self._pending_stop_loss = (
            float(self._opening_low) if signal == "BUY" else float(self._opening_high)
        )
        self._trade_taken = True
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        stop_loss = self._pending_stop_loss
        self._pending_stop_loss = None
        if self.fixed_stop_loss_pct is not None and self.fixed_stop_loss_pct > 0:
            close = float(row.get("close", float("nan")))
            if pd.isna(close) or close <= 0:
                return None
            if signal == "BUY":
                return close * (1.0 - self.fixed_stop_loss_pct)
            if signal == "SHORT":
                return close * (1.0 + self.fixed_stop_loss_pct)
        return stop_loss


@dataclass
class FirstFiveMinuteFakeBreakoutStrategy:
    """Fade failed breakouts of the first five-minute candle using one-minute execution."""

    opening_start: time = time(hour=9, minute=15)
    breakout_start: time = time(hour=9, minute=20)
    failure_deadline: time = time(hour=9, minute=40)
    trade_deadline: time = time(hour=10, minute=0)
    risk_reward_multiple: float = 1.0
    fixed_stop_loss_pct: float | None = None
    fixed_take_profit_pct: float | None = None
    use_volume_filter: bool = True
    use_vwap_filter: bool = True
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _opening_high: float | None = field(default=None, init=False, repr=False)
    _opening_low: float | None = field(default=None, init=False, repr=False)
    _opening_ready: bool = field(default=False, init=False, repr=False)
    _breakout_up_seen: bool = field(default=False, init=False, repr=False)
    _breakout_down_seen: bool = field(default=False, init=False, repr=False)
    _breakout_up_extreme: float | None = field(default=None, init=False, repr=False)
    _breakout_down_extreme: float | None = field(default=None, init=False, repr=False)
    _trade_taken: bool = field(default=False, init=False, repr=False)
    _pending_stop_loss: float | None = field(default=None, init=False, repr=False)

    required_columns: tuple[str, ...] = ("volume", "rolling_volume_avg", "vwap")

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._opening_high = None
        self._opening_low = None
        self._opening_ready = False
        self._breakout_up_seen = False
        self._breakout_down_seen = False
        self._breakout_up_extreme = None
        self._breakout_down_extreme = None
        self._trade_taken = False
        self._pending_stop_loss = None

    def _update_opening_candle(self, row: pd.Series, candle_time: time) -> None:
        if not (self.opening_start <= candle_time < self.breakout_start):
            if candle_time >= self.breakout_start and self._opening_high is not None:
                self._opening_ready = True
            return

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        if pd.isna(high) or pd.isna(low):
            return

        self._opening_high = high if self._opening_high is None else max(self._opening_high, high)
        self._opening_low = low if self._opening_low is None else min(self._opening_low, low)

    def _entry_filters_ok(self, row: pd.Series, signal: Signal) -> bool:
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        close = float(row.get("close", float("nan")))

        if self.use_volume_filter:
            if pd.isna(volume) or pd.isna(volume_avg) or volume_avg <= 0 or volume <= volume_avg:
                return False

        if self.use_vwap_filter:
            if pd.isna(vwap) or pd.isna(close):
                return False
            if signal == "BUY" and close <= vwap:
                return False
            if signal == "SHORT" and close >= vwap:
                return False

        return True

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        self._update_opening_candle(row, candle_time)

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if context.is_end_of_day or candle_time > self.trade_deadline:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"

            if context.position_side == "LONG":
                if self.fixed_take_profit_pct is not None and self.fixed_take_profit_pct > 0:
                    target = entry_price * (1.0 + self.fixed_take_profit_pct)
                else:
                    target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"

            if self.fixed_take_profit_pct is not None and self.fixed_take_profit_pct > 0:
                target = entry_price * (1.0 - self.fixed_take_profit_pct)
            else:
                target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        if (
            not self._opening_ready
            or self._trade_taken
            or self._opening_high is None
            or self._opening_low is None
            or candle_time < self.breakout_start
            or candle_time > self.trade_deadline
        ):
            return "HOLD"

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            return "HOLD"

        if candle_time <= self.failure_deadline:
            if high > self._opening_high:
                self._breakout_up_seen = True
                self._breakout_up_extreme = (
                    high
                    if self._breakout_up_extreme is None
                    else max(self._breakout_up_extreme, high)
                )
            if low < self._opening_low:
                self._breakout_down_seen = True
                self._breakout_down_extreme = (
                    low
                    if self._breakout_down_extreme is None
                    else min(self._breakout_down_extreme, low)
                )

        signal: Signal = "HOLD"
        if self._breakout_up_seen and close < self._opening_high and self.allow_shorts:
            signal = "SHORT"
            self._pending_stop_loss = (
                self._breakout_up_extreme
                if self._breakout_up_extreme is not None
                else self._opening_high
            )
        elif self._breakout_down_seen and close > self._opening_low:
            signal = "BUY"
            self._pending_stop_loss = (
                self._breakout_down_extreme
                if self._breakout_down_extreme is not None
                else self._opening_low
            )

        if signal == "HOLD" or not self._entry_filters_ok(row, signal):
            if signal == "HOLD":
                self._pending_stop_loss = None
            return "HOLD"

        if self.reverse_signals:
            signal = reverse_signal(signal)
            self._pending_stop_loss = (
                float(self._opening_low) if signal == "BUY" else float(self._opening_high)
            )

        self._trade_taken = True
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        stop_loss = self._pending_stop_loss
        self._pending_stop_loss = None
        if self.fixed_stop_loss_pct is not None and self.fixed_stop_loss_pct > 0:
            close = float(row.get("close", float("nan")))
            if pd.isna(close) or close <= 0:
                return None
            if signal == "BUY":
                return close * (1.0 - self.fixed_stop_loss_pct)
            if signal == "SHORT":
                return close * (1.0 + self.fixed_stop_loss_pct)
        return stop_loss


@dataclass
class InsideBarBreakoutStrategy:
    """Inside bar breakout strategy with 1:1 RR and optional filters."""

    entry_session_start: time = time(hour=9, minute=15)
    entry_session_end: time = time(hour=15, minute=15)
    session_exit_time: time = time(hour=15, minute=15)
    max_setup_candles: int = 0
    max_holding_candles_without_profit: int = 0
    min_mother_range_pct: float = 0.0015
    max_stop_loss_pct: float = 0.0
    min_distance_from_open: float = 0.0
    use_atr_stop: bool = False
    atr_stop_multiple: float = 1.0
    risk_reward_multiple: float = 1.0
    use_inside_bar_range: bool = False
    probability_threshold: float = 0.0
    buy_probability_column: str = "buy_probability"
    sell_probability_column: str = "sell_probability"
    use_volume_filter: bool = False
    use_vwap_trend_filter: bool = False
    use_ema_trend_filter: bool = False
    allow_longs: bool = True
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _bar_index: int = field(default=0, init=False, repr=False)
    _setup_active: bool = field(default=False, init=False, repr=False)
    _setup_expires_at: int | None = field(default=None, init=False, repr=False)
    _mother_high: float | None = field(default=None, init=False, repr=False)
    _mother_low: float | None = field(default=None, init=False, repr=False)
    _prev_high: float | None = field(default=None, init=False, repr=False)
    _prev_low: float | None = field(default=None, init=False, repr=False)
    _clear_after_entry: bool = field(default=False, init=False, repr=False)
    _position_entry_bar: int | None = field(default=None, init=False, repr=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = ["high", "low", "close", "volume"]
        if self.min_distance_from_open > 0:
            columns.append("distance_from_open")
        if self.use_volume_filter:
            columns.append("rolling_volume_avg")
        if self.use_vwap_trend_filter:
            columns.append("vwap")
        if self.use_ema_trend_filter:
            columns.extend(["ema20", "ema50", "ema200"])
        if self.use_atr_stop:
            columns.append("atr")
        if self.probability_threshold and self.probability_threshold > 0:
            columns.extend([self.buy_probability_column, self.sell_probability_column])
        return tuple(dict.fromkeys(columns))

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._bar_index = 0
        self._setup_active = False
        self._setup_expires_at = None
        self._mother_high = None
        self._mother_low = None
        self._prev_high = None
        self._prev_low = None
        self._position_entry_bar = None

    def _in_session(self, candle_time: time) -> bool:
        return self.entry_session_start <= candle_time <= self.entry_session_end

    def _volume_ok(self, row: pd.Series) -> bool:
        if not self.use_volume_filter:
            return True
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        if pd.isna(volume) or pd.isna(volume_avg):
            return False
        return volume_avg > 0 and volume > volume_avg

    def _trend_ok(self, row: pd.Series, side: Signal) -> bool:
        if not self.use_vwap_trend_filter and not self.use_ema_trend_filter:
            return True
        close = float(row.get("close", float("nan")))
        if pd.isna(close):
            return False

        if self.use_vwap_trend_filter:
            vwap = float(row.get("vwap", float("nan")))
            if pd.isna(vwap):
                return False
            if side == "BUY" and close <= vwap:
                return False
            if side == "SELL" and close >= vwap:
                return False

        if self.use_ema_trend_filter:
            ema20 = float(row.get("ema20", float("nan")))
            ema50 = float(row.get("ema50", float("nan")))
            ema200 = float(row.get("ema200", float("nan")))
            if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ema200):
                return False
            if side == "BUY" and not (ema20 > ema50 > ema200):
                return False
            if side == "SELL" and not (ema20 < ema50 < ema200):
                return False

        return True

    def _probability_ok(self, row: pd.Series, side: Signal) -> bool:
        if not self.probability_threshold or self.probability_threshold <= 0:
            return True
        column = (
            self.buy_probability_column if side == "BUY" else self.sell_probability_column
        )
        proba = float(row.get(column, float("nan")))
        if pd.isna(proba):
            return False
        return proba >= self.probability_threshold

    def _mother_range_ok(self, mother_high: float, mother_low: float, close: float) -> bool:
        if close <= 0:
            return False
        return (mother_high - mother_low) / close > self.min_mother_range_pct

    def _open_distance_ok(self, row: pd.Series, side: Signal) -> bool:
        if self.min_distance_from_open <= 0:
            return True
        distance_from_open = float(row.get("distance_from_open", float("nan")))
        if pd.isna(distance_from_open):
            return False
        if side == "BUY":
            return distance_from_open >= self.min_distance_from_open
        if side == "SELL":
            return distance_from_open <= -self.min_distance_from_open
        return True

    def _store_setup(self, mother_high: float, mother_low: float) -> None:
        self._setup_active = True
        self._mother_high = mother_high
        self._mother_low = mother_low
        if self.max_setup_candles > 0:
            self._setup_expires_at = self._bar_index + self.max_setup_candles
        else:
            self._setup_expires_at = None

    def _clear_setup(self) -> None:
        self._setup_active = False
        self._setup_expires_at = None
        self._mother_high = None
        self._mother_low = None
        self._clear_after_entry = False

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        self._bar_index += 1

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if candle_time >= self.session_exit_time:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            close = float(row.get("close", float("nan")))
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                return "HOLD"

            if (
                self.max_holding_candles_without_profit > 0
                and self._position_entry_bar is not None
                and (self._bar_index - self._position_entry_bar)
                >= self.max_holding_candles_without_profit
            ):
                if context.position_side == "LONG" and close <= entry_price:
                    return "SELL"
                if context.position_side == "SHORT" and close >= entry_price:
                    return "COVER"

            if context.position_side == "LONG":
                target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"
            target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        self._position_entry_bar = None

        if not self._in_session(candle_time):
            self._clear_setup()
            return "HOLD"

        if self._setup_active and self._setup_expires_at is not None:
            if self._bar_index > self._setup_expires_at:
                self._clear_setup()

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            return "HOLD"

        if self._setup_active and self._mother_high is not None and self._mother_low is not None:
            breakout_high = self._mother_high
            breakout_low = self._mother_low
            if (
                high > breakout_high
                and self.allow_longs
                and self._open_distance_ok(row, "BUY")
                and self._volume_ok(row)
                and self._trend_ok(row, "BUY")
                and self._probability_ok(row, "BUY")
            ):
                self._setup_active = False
                self._clear_after_entry = True
                self._position_entry_bar = self._bar_index
                return "SHORT" if self.reverse_signals else "BUY"
            if (
                low < breakout_low
                and self.allow_shorts
                and self._open_distance_ok(row, "SELL")
                and self._volume_ok(row)
                and self._trend_ok(row, "SELL")
                and self._probability_ok(row, "SELL")
            ):
                self._setup_active = False
                self._clear_after_entry = True
                self._position_entry_bar = self._bar_index
                return "BUY" if self.reverse_signals else "SHORT"
            return "HOLD"

        if self._prev_high is None or self._prev_low is None:
            self._prev_high = high
            self._prev_low = low
            return "HOLD"

        inside_bar = high <= self._prev_high and low >= self._prev_low
        if inside_bar:
            if self.use_inside_bar_range:
                range_high = high
                range_low = low
            else:
                range_high = self._prev_high
                range_low = self._prev_low

            if self._mother_range_ok(range_high, range_low, close):
                self._store_setup(range_high, range_low)

        self._prev_high = high
        self._prev_low = low

        return "HOLD"

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        if self._mother_high is None or self._mother_low is None:
            return None
        if signal == "BUY":
            if self.use_atr_stop:
                atr = float(row.get("atr", float("nan")))
                close = float(row.get("close", float("nan")))
                if pd.isna(atr) or pd.isna(close) or atr <= 0:
                    return None
                stop = close - (atr * self.atr_stop_multiple)
            else:
                stop = self._mother_low
        elif signal == "SHORT":
            if self.use_atr_stop:
                atr = float(row.get("atr", float("nan")))
                close = float(row.get("close", float("nan")))
                if pd.isna(atr) or pd.isna(close) or atr <= 0:
                    return None
                stop = close + (atr * self.atr_stop_multiple)
            else:
                stop = self._mother_high
        else:
            stop = None

        if stop is not None and self.max_stop_loss_pct > 0:
            close = float(row.get("close", float("nan")))
            if pd.isna(close) or close <= 0:
                return None
            stop_distance_pct = abs(close - stop) / close
            if stop_distance_pct > self.max_stop_loss_pct:
                if signal == "BUY":
                    stop = close * (1.0 - self.max_stop_loss_pct)
                elif signal == "SHORT":
                    stop = close * (1.0 + self.max_stop_loss_pct)

        if self._clear_after_entry:
            self._clear_setup()
        return stop


@dataclass
class InsideBarFakeBreakoutReversalStrategy:
    """Trade the reversal after an inside-bar breakout fails back into the mother range."""

    entry_session_start: time = time(hour=9, minute=15)
    entry_session_end: time = time(hour=15, minute=15)
    session_exit_time: time = time(hour=15, minute=15)
    risk_reward_multiple: float = 1.0
    max_stop_loss_pct: float = 0.006
    allow_longs: bool = True
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _mother_high: float | None = field(default=None, init=False, repr=False)
    _mother_low: float | None = field(default=None, init=False, repr=False)
    _prev_high: float | None = field(default=None, init=False, repr=False)
    _prev_low: float | None = field(default=None, init=False, repr=False)
    _breakout_side: Signal | None = field(default=None, init=False, repr=False)
    _breakout_extreme: float | None = field(default=None, init=False, repr=False)
    _pending_stop_loss: float | None = field(default=None, init=False, repr=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("high", "low", "close")

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._mother_high = None
        self._mother_low = None
        self._prev_high = None
        self._prev_low = None
        self._breakout_side = None
        self._breakout_extreme = None
        self._pending_stop_loss = None

    def _clear_setup(self) -> None:
        self._mother_high = None
        self._mother_low = None
        self._breakout_side = None
        self._breakout_extreme = None
        self._pending_stop_loss = None

    def _clear_setup_keep_pending_stop(self) -> None:
        self._mother_high = None
        self._mother_low = None
        self._breakout_side = None
        self._breakout_extreme = None

    def _in_entry_session(self, candle_time: time) -> bool:
        return self.entry_session_start <= candle_time <= self.entry_session_end

    def _capped_stop(self, close: float, stop: float, signal: Signal) -> float:
        if self.max_stop_loss_pct <= 0 or close <= 0:
            return stop
        stop_distance_pct = abs(close - stop) / close
        if stop_distance_pct <= self.max_stop_loss_pct:
            return stop
        if signal == "BUY":
            return close * (1.0 - self.max_stop_loss_pct)
        if signal == "SHORT":
            return close * (1.0 + self.max_stop_loss_pct)
        return stop

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            return "HOLD"

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if candle_time >= self.session_exit_time:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            if context.position_side == "LONG":
                target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"
            target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        if not self._in_entry_session(candle_time):
            return "HOLD"

        if self._mother_high is not None and self._mother_low is not None:
            if self._breakout_side is None:
                if high > self._mother_high:
                    self._breakout_side = "BUY"
                    self._breakout_extreme = high
                elif low < self._mother_low:
                    self._breakout_side = "SHORT"
                    self._breakout_extreme = low
            elif self._breakout_side == "BUY":
                self._breakout_extreme = (
                    high
                    if self._breakout_extreme is None
                    else max(self._breakout_extreme, high)
                )
                if close < self._mother_high and close >= self._mother_low and self.allow_shorts:
                    self._pending_stop_loss = self._capped_stop(close, float(self._breakout_extreme), "SHORT")
                    self._clear_setup_keep_pending_stop()
                    return "BUY" if self.reverse_signals else "SHORT"
            elif self._breakout_side == "SHORT":
                self._breakout_extreme = (
                    low
                    if self._breakout_extreme is None
                    else min(self._breakout_extreme, low)
                )
                if close > self._mother_low and close <= self._mother_high and self.allow_longs:
                    self._pending_stop_loss = self._capped_stop(close, float(self._breakout_extreme), "BUY")
                    self._clear_setup_keep_pending_stop()
                    return "SHORT" if self.reverse_signals else "BUY"

        if self._prev_high is None or self._prev_low is None:
            self._prev_high = high
            self._prev_low = low
            return "HOLD"

        inside_bar = high <= self._prev_high and low >= self._prev_low
        if inside_bar:
            self._mother_high = self._prev_high
            self._mother_low = self._prev_low
            self._breakout_side = None
            self._breakout_extreme = None

        self._prev_high = high
        self._prev_low = low
        return "HOLD"

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del row, signal, stop_atr_multiple
        stop_loss = self._pending_stop_loss
        self._pending_stop_loss = None
        return stop_loss


@dataclass
class SupportResistanceLevel:
    level_type: Literal["support", "resistance"]
    price: float
    touch_count: int
    created_index: int
    last_touched_index: int
    last_trade_index: int | None = None


@dataclass
class SupportResistanceReversalStrategy:
    """Intraday support/resistance reversal strategy with swing-level detection."""

    entry_session_start: time = time(hour=9, minute=20)
    entry_session_end: time = time(hour=14, minute=30)
    zone_tolerance_pct: float = 0.0005
    distance_threshold_pct: float = 0.0008
    stop_offset_pct: float = 0.001
    risk_reward_multiple: float = 1.0
    cooldown_candles: int = 10
    level_expiry_candles: int = 120
    min_touch_count: int = 2
    volume_multiplier: float = 1.0
    use_volume_filter: bool = True
    use_vwap_filter: bool = True
    use_trend_filter: bool = True
    ema20_slope_threshold_pct: float = 0.001
    use_external_levels: bool = False
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _bar_index: int = field(default=0, init=False, repr=False)
    _levels: list[SupportResistanceLevel] = field(default_factory=list, init=False, repr=False)
    _buffer: deque[dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=5), init=False, repr=False
    )
    _pending_entry_level_price: float | None = field(default=None, init=False, repr=False)
    _pending_entry_signal: Signal | None = field(default=None, init=False, repr=False)
    _prev_ema20: float | None = field(default=None, init=False, repr=False)
    _external_level_last_trade: dict[tuple[str, float], int] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = ["volume", "rolling_volume_avg", "vwap", "ema20"]
        if self.use_external_levels:
            columns.extend(
                [
                    "sr_support_price",
                    "sr_support_touch_count",
                    "sr_resistance_price",
                    "sr_resistance_touch_count",
                ]
            )
        return tuple(columns)

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._bar_index = 0
        self._levels = []
        self._buffer.clear()
        self._pending_entry_level_price = None
        self._pending_entry_signal = None
        self._prev_ema20 = None
        self._external_level_last_trade = {}

    def _in_session(self, candle_time: time) -> bool:
        return self.entry_session_start <= candle_time <= self.entry_session_end

    def _update_swing_levels(self) -> None:
        if len(self._buffer) < 5:
            return
        highs = [candle["high"] for candle in self._buffer]
        lows = [candle["low"] for candle in self._buffer]

        center_high = highs[2]
        center_low = lows[2]
        center_index = self._bar_index - 2

        if center_high > highs[0] and center_high > highs[1] and center_high > highs[3] and center_high > highs[4]:
            self._add_level("resistance", center_high, center_index)

        if center_low < lows[0] and center_low < lows[1] and center_low < lows[3] and center_low < lows[4]:
            self._add_level("support", center_low, center_index)

    def _add_level(self, level_type: Literal["support", "resistance"], price: float, index: int) -> None:
        if price <= 0:
            return
        for level in self._levels:
            if level.level_type != level_type:
                continue
            if self._within_zone(price, level.price):
                combined_touches = level.touch_count + 1
                level.price = (level.price * level.touch_count + price) / combined_touches
                level.touch_count = combined_touches
                level.last_touched_index = index
                return

        self._levels.append(
            SupportResistanceLevel(
                level_type=level_type,
                price=price,
                touch_count=1,
                created_index=index,
                last_touched_index=index,
            )
        )

    def _within_zone(self, price: float, level_price: float) -> bool:
        if level_price <= 0:
            return False
        return abs(price - level_price) / level_price <= self.zone_tolerance_pct

    def _update_level_touches(self, low: float, high: float) -> None:
        for level in self._levels:
            touch_price = low if level.level_type == "support" else high
            if self._within_zone(touch_price, level.price):
                if level.last_touched_index != self._bar_index:
                    level.touch_count += 1
                    level.last_touched_index = self._bar_index

    def _expire_levels(self) -> None:
        expiry = max(self.level_expiry_candles, 1)
        self._levels = [
            level
            for level in self._levels
            if (self._bar_index - level.last_touched_index) <= expiry
        ]

    def _volume_ok(self, volume: float, volume_avg: float) -> bool:
        if not self.use_volume_filter:
            return True
        if volume_avg <= 0:
            return False
        return volume > (self.volume_multiplier * volume_avg)

    def _trend_ok(self, ema20: float, close: float) -> bool:
        if not self.use_trend_filter:
            return True
        if self._prev_ema20 is None or close <= 0:
            return True
        slope_pct = abs(ema20 - self._prev_ema20) / close
        return slope_pct <= self.ema20_slope_threshold_pct

    @staticmethod
    def _bullish_rejection(open_price: float, close: float, low: float) -> bool:
        body = abs(close - open_price)
        lower_wick = min(open_price, close) - low
        return close > open_price and lower_wick > body

    @staticmethod
    def _bearish_rejection(open_price: float, close: float, high: float) -> bool:
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        return close < open_price and upper_wick > body

    def _level_strength(self, level: SupportResistanceLevel) -> float:
        age = max(self._bar_index - level.last_touched_index, 0)
        expiry = max(self.level_expiry_candles, 1)
        recency_bonus = max(0.0, (expiry - age) / expiry)
        return (level.touch_count * 2.0) + recency_bonus

    def _select_level(
        self, level_type: Literal["support", "resistance"], touch_price: float
    ) -> SupportResistanceLevel | None:
        if touch_price <= 0:
            return None
        candidates: list[SupportResistanceLevel] = []
        for level in self._levels:
            if level.level_type != level_type:
                continue
            if level.touch_count < self.min_touch_count:
                continue
            if level.last_trade_index is not None:
                if (self._bar_index - level.last_trade_index) <= self.cooldown_candles:
                    continue
            distance = abs(touch_price - level.price) / level.price
            if distance <= self.distance_threshold_pct:
                candidates.append(level)
        if not candidates:
            return None
        return max(candidates, key=self._level_strength)

    def _external_level_key(
        self, level_type: Literal["support", "resistance"], price: float
    ) -> tuple[str, float]:
        return (level_type, round(price, 6))

    def _select_external_level(
        self, row: pd.Series, level_type: Literal["support", "resistance"], touch_price: float
    ) -> SupportResistanceLevel | None:
        price = float(row.get(f"sr_{level_type}_price", float("nan")))
        touch_count = float(row.get(f"sr_{level_type}_touch_count", float("nan")))
        if pd.isna(price) or pd.isna(touch_count) or price <= 0:
            return None
        if touch_count < self.min_touch_count:
            return None
        distance = abs(touch_price - price) / price
        if distance > self.distance_threshold_pct:
            return None
        level_key = self._external_level_key(level_type, price)
        last_trade_index = self._external_level_last_trade.get(level_key)
        if last_trade_index is not None and (
            self._bar_index - last_trade_index
        ) <= self.cooldown_candles:
            return None
        return SupportResistanceLevel(
            level_type=level_type,
            price=price,
            touch_count=int(touch_count),
            created_index=self._bar_index,
            last_touched_index=self._bar_index,
        )

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        open_price = float(row.get("open", float("nan")))
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        ema20 = float(row.get("ema20", float("nan")))
        inputs = (open_price, high, low, close, volume, volume_avg, vwap, ema20)
        if any(pd.isna(v) for v in inputs):
            return "HOLD"

        self._bar_index += 1
        if not self.use_external_levels:
            self._buffer.append({"high": high, "low": low})
            self._update_swing_levels()
            self._update_level_touches(low=low, high=high)
            self._expire_levels()

        signal: Signal = "HOLD"
        if context.in_position:
            if context.position_side is None:
                signal = "HOLD"
            elif context.is_end_of_day:
                signal = "SELL" if context.position_side == "LONG" else "COVER"
            elif context.position_entry_price is None or context.position_stop_loss is None:
                signal = "HOLD"
            else:
                entry_price = context.position_entry_price
                stop_loss = context.position_stop_loss
                risk = abs(entry_price - stop_loss)
                if risk <= 0:
                    signal = "HOLD"
                elif context.position_side == "LONG":
                    target = entry_price + (risk * self.risk_reward_multiple)
                    signal = "SELL" if high >= target else "HOLD"
                else:
                    target = entry_price - (risk * self.risk_reward_multiple)
                    signal = "COVER" if low <= target else "HOLD"
        else:
            if not self._in_session(candle_time):
                self._prev_ema20 = ema20
                return "HOLD"

            if not self._volume_ok(volume=volume, volume_avg=volume_avg):
                self._prev_ema20 = ema20
                return "HOLD"

            if not self._trend_ok(ema20=ema20, close=close):
                self._prev_ema20 = ema20
                return "HOLD"

            long_ok = (not self.use_vwap_filter) or (close > vwap)
            short_ok = (not self.use_vwap_filter) or (close < vwap)

            if self.use_external_levels:
                support_level = self._select_external_level(
                    row=row, level_type="support", touch_price=low
                )
                resistance_level = self._select_external_level(
                    row=row, level_type="resistance", touch_price=high
                )
            else:
                support_level = self._select_level("support", touch_price=low)
                resistance_level = self._select_level("resistance", touch_price=high)

            if support_level and long_ok and self._bullish_rejection(open_price, close, low):
                signal = "BUY"
                self._pending_entry_level_price = support_level.price
                if self.use_external_levels:
                    level_key = self._external_level_key("support", support_level.price)
                    self._external_level_last_trade[level_key] = self._bar_index
                else:
                    support_level.last_trade_index = self._bar_index

            if (
                signal == "HOLD"
                and resistance_level
                and short_ok
                and self.allow_shorts
                and self._bearish_rejection(open_price, close, high)
            ):
                signal = "SHORT"
                self._pending_entry_level_price = resistance_level.price
                if self.use_external_levels:
                    level_key = self._external_level_key("resistance", resistance_level.price)
                    self._external_level_last_trade[level_key] = self._bar_index
                else:
                    resistance_level.last_trade_index = self._bar_index

        self._prev_ema20 = ema20

        if self.reverse_signals:
            signal = reverse_signal(signal)

        if signal in {"BUY", "SHORT"}:
            self._pending_entry_signal = signal
        else:
            self._pending_entry_signal = None
            if signal != "HOLD":
                self._pending_entry_level_price = None

        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        level_price = self._pending_entry_level_price
        pending_signal = self._pending_entry_signal
        self._pending_entry_level_price = None
        self._pending_entry_signal = None
        if level_price is None or pending_signal is None:
            return None

        if pending_signal == "BUY":
            return level_price * (1.0 - self.stop_offset_pct)
        if pending_signal == "SHORT":
            return level_price * (1.0 + self.stop_offset_pct)
        return None


@dataclass
class RandomOpenDirectionStrategy:
    """Enter a deterministic random long/short trade on a specific intraday candle."""

    entry_time: time = time(hour=9, minute=15)
    risk_reward_multiple: float = 1.0
    seed: int = 42
    allow_shorts: bool = True
    reverse_signals: bool = False
    _current_day: object | None = field(default=None, init=False, repr=False)
    _day_has_traded: bool = field(default=False, init=False, repr=False)
    _pending_entry_signal: Signal | None = field(default=None, init=False, repr=False)

    required_columns: tuple[str, ...] = ()

    def _reset_day(self, day: object) -> None:
        self._current_day = day
        self._day_has_traded = False
        self._pending_entry_signal = None

    def _pick_direction(self, candle_day: object) -> Signal:
        day_seed = f"{self.seed}-{candle_day}"
        chooser = random.Random(day_seed)
        if self.allow_shorts:
            signal = chooser.choice(["BUY", "SHORT"])
        else:
            signal = "BUY"
        return reverse_signal(signal) if self.reverse_signals else signal

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        if context.in_position:
            if context.position_side is None:
                return "HOLD"
            if context.is_end_of_day:
                return "SELL" if context.position_side == "LONG" else "COVER"
            if context.position_entry_price is None or context.position_stop_loss is None:
                return "HOLD"

            entry_price = context.position_entry_price
            stop_loss = context.position_stop_loss
            risk = abs(entry_price - stop_loss)
            if risk <= 0:
                return "HOLD"

            high = float(row.get("high", float("nan")))
            low = float(row.get("low", float("nan")))
            if pd.isna(high) or pd.isna(low):
                return "HOLD"

            if context.position_side == "LONG":
                target = entry_price + (risk * self.risk_reward_multiple)
                return "SELL" if high >= target else "HOLD"

            target = entry_price - (risk * self.risk_reward_multiple)
            return "COVER" if low <= target else "HOLD"

        if self._day_has_traded or candle_time != self.entry_time:
            return "HOLD"

        signal = self._pick_direction(candle_day)
        self._pending_entry_signal = signal
        self._day_has_traded = True
        return signal

    def entry_stop_loss(
        self,
        row: pd.Series,
        signal: Signal,
        stop_atr_multiple: float,
    ) -> float | None:
        del stop_atr_multiple
        pending_signal = self._pending_entry_signal
        self._pending_entry_signal = None
        if pending_signal is None:
            return None

        low = float(row.get("low", float("nan")))
        high = float(row.get("high", float("nan")))
        close = float(row.get("close", float("nan")))
        if pd.isna(low) or pd.isna(high) or pd.isna(close):
            return None

        if pending_signal == "BUY":
            return low if low < close else None
        if pending_signal == "SHORT":
            return high if high > close else None
        return None

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


@dataclass(frozen=True)
class OneMinuteVwapEma9IciciFocusedStrategy:
    """ICICI-focused 1-minute scalp variant with stricter filters."""

    required_columns: tuple[str, ...] = (
        "vwap",
        "ema9",
        "rolling_volume_avg",
        "volume",
        "atr",
        "bb_width",
    )
    allow_shorts: bool = True
    reverse_signals: bool = False
    volume_spike_multiplier: float = 1.8
    stop_min_pct: float = 0.002
    stop_max_pct: float = 0.0025
    risk_reward_multiple: float = 1.5
    take_profit_mode: Literal["rr", "atr"] = "rr"
    min_bb_width: float = 0.004
    min_atr_pct: float = 0.001
    max_ema_distance_atr: float = 0.4
    session_start: time = time(hour=9, minute=20)
    session_end: time = time(hour=14, minute=45)

    def _in_session(self, row: pd.Series) -> bool:
        ts = row.get("timestamp")
        if ts is None:
            return False
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return False
        t = timestamp.time()
        return self.session_start <= t <= self.session_end

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        if not self._in_session(row):
            if context.in_position and context.is_end_of_day:
                return "SELL" if context.position_side == "LONG" else "COVER"
            return "HOLD"

        close = float(row.get("close", float("nan")))
        open_price = float(row.get("open", float("nan")))
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        ema9 = float(row.get("ema9", float("nan")))
        atr = float(row.get("atr", float("nan")))
        bb_width = float(row.get("bb_width", float("nan")))
        volume = float(row.get("volume", float("nan")))
        volume_avg = float(row.get("rolling_volume_avg", float("nan")))
        inputs = (close, open_price, high, low, vwap, ema9, atr, bb_width, volume, volume_avg)

        if any(pd.isna(v) for v in inputs):
            return "HOLD"
        if close <= 0 or volume_avg <= 0 or atr <= 0:
            return "HOLD"

        # No-trade chop filter.
        if bb_width < self.min_bb_width or (atr / close) < self.min_atr_pct:
            return "HOLD"

        signal: Signal = "HOLD"
        if context.in_position:
            signal = self._exit_signal(close=close, atr=atr, context=context)
        else:
            near_ema = abs(close - ema9) <= (self.max_ema_distance_atr * atr)
            long_setup = (
                close > vwap
                and low <= ema9
                and close > ema9
                and close > open_price
                and near_ema
                and volume > (self.volume_spike_multiplier * volume_avg)
            )
            short_setup = (
                close < vwap
                and high >= ema9
                and close < ema9
                and close < open_price
                and near_ema
                and volume > (self.volume_spike_multiplier * volume_avg)
            )
            if long_setup:
                signal = "BUY"
            elif self.allow_shorts and short_setup:
                signal = "SHORT"

        return reverse_signal(signal) if self.reverse_signals else signal

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
            target = (
                entry_price + atr
                if self.take_profit_mode == "atr"
                else entry_price + (self.risk_reward_multiple * risk)
            )
            return "SELL" if close >= target else "HOLD"

        risk = stop_loss - entry_price
        if risk <= 0:
            return "HOLD"
        target = (
            entry_price - atr
            if self.take_profit_mode == "atr"
            else entry_price - (self.risk_reward_multiple * risk)
        )
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
        if any(pd.isna(v) for v in (close, low, high)):
            return None
        if close <= 0:
            return None

        min_distance = close * self.stop_min_pct
        max_distance = close * self.stop_max_pct
        if signal == "BUY":
            swing_distance = max(close - low, 0.0)
            distance = min(max(swing_distance, min_distance), max_distance)
            return close - distance
        if signal == "SHORT":
            swing_distance = max(high - close, 0.0)
            distance = min(max(swing_distance, min_distance), max_distance)
            return close + distance
        return None
