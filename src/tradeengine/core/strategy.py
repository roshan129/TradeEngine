from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class StrategyContext:
    """Per-candle execution context passed into stateless strategies."""

    in_position: bool
    available_capital: float
    is_end_of_day: bool


class Strategy(Protocol):
    """Strategy contract for deterministic, row-wise signal generation."""

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        """Return one of BUY, SELL, or HOLD for the current candle."""


@dataclass(frozen=True)
class BaselineEmaRsiStrategy:
    """Baseline long-only strategy used to validate backtesting plumbing."""

    entry_rsi_threshold: float = 55.0
    exit_rsi_threshold: float = 45.0

    def generate_signal(self, row: pd.Series, context: StrategyContext) -> Signal:
        ema20 = float(row.get("ema20", float("nan")))
        ema50 = float(row.get("ema50", float("nan")))
        rsi = float(row.get("rsi", float("nan")))

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(rsi):
            return "HOLD"

        if context.in_position:
            if rsi < self.exit_rsi_threshold or context.is_end_of_day:
                return "SELL"
            return "HOLD"

        if ema20 > ema50 and rsi > self.entry_rsi_threshold:
            return "BUY"

        return "HOLD"
