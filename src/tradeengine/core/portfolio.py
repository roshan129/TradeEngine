from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class CostModel:
    """Transaction cost model with fixed and percentage brokerage plus slippage."""

    brokerage_fixed: float = 20.0
    brokerage_pct: float = 0.0003
    slippage_pct: float = 0.0005

    def brokerage(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        fixed = max(self.brokerage_fixed, 0.0)
        variable = max(self.brokerage_pct, 0.0) * notional
        return max(fixed, variable)


@dataclass
class Position:
    side: PositionSide
    entry_price: float
    quantity: int
    stop_loss: float
    entry_time: pd.Timestamp
    entry_index: int
    entry_brokerage: float
    risk_amount: float


@dataclass(frozen=True)
class TradeRecord:
    side: PositionSide
    entry_timestamp: pd.Timestamp
    entry_price: float
    exit_timestamp: pd.Timestamp
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    r_multiple: float
    trade_duration_minutes: float
    exit_reason: str


class PortfolioError(ValueError):
    """Raised when invalid portfolio operations are attempted."""


class Portfolio:
    """Single-position portfolio model with deterministic trade accounting."""

    def __init__(self, initial_capital: float, cost_model: CostModel) -> None:
        if initial_capital <= 0:
            raise PortfolioError("initial_capital must be positive")

        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.cost_model = cost_model
        self.open_position: Position | None = None
        self.trade_log: list[TradeRecord] = []

    @property
    def has_open_position(self) -> bool:
        return self.open_position is not None

    def estimate_entry_fill_price(self, price: float, side: PositionSide) -> float:
        if side == "LONG":
            return float(price) * (1.0 + self.cost_model.slippage_pct)
        return float(price) * (1.0 - self.cost_model.slippage_pct)

    def estimate_exit_fill_price(self, price: float, side: PositionSide) -> float:
        if side == "LONG":
            return float(price) * (1.0 - self.cost_model.slippage_pct)
        return float(price) * (1.0 + self.cost_model.slippage_pct)

    def enter_position(
        self,
        side: PositionSide,
        timestamp: pd.Timestamp,
        candle_close: float,
        stop_loss: float,
        quantity: int,
        entry_index: int,
        risk_amount: float,
    ) -> bool:
        if self.has_open_position:
            return False
        if quantity <= 0:
            return False

        entry_price = self.estimate_entry_fill_price(candle_close, side=side)
        notional = entry_price * quantity
        brokerage = self.cost_model.brokerage(notional)
        if side == "LONG":
            total_cost = notional + brokerage
            if total_cost > self.capital:
                return False
            self.capital -= total_cost
        else:
            # Short entry receives sale proceeds; brokerage is paid at entry.
            self.capital += notional - brokerage

        self.open_position = Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=float(stop_loss),
            entry_time=timestamp,
            entry_index=entry_index,
            entry_brokerage=brokerage,
            risk_amount=max(risk_amount, 0.0),
        )
        return True

    def exit_position(
        self,
        timestamp: pd.Timestamp,
        candle_price: float,
        exit_reason: str,
        use_stop_fill: bool = False,
    ) -> TradeRecord:
        position = self.open_position
        if position is None:
            raise PortfolioError("No open position to exit")

        raw_exit_price = position.stop_loss if use_stop_fill else float(candle_price)
        exit_price = self.estimate_exit_fill_price(raw_exit_price, side=position.side)
        exit_notional = exit_price * position.quantity
        exit_brokerage = self.cost_model.brokerage(exit_notional)

        if position.side == "LONG":
            gross_pnl = (exit_price - position.entry_price) * position.quantity
            self.capital += exit_notional - exit_brokerage
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
            self.capital -= exit_notional + exit_brokerage

        net_pnl = gross_pnl - position.entry_brokerage - exit_brokerage

        trade_duration = timestamp - position.entry_time
        trade_duration_minutes = float(trade_duration.total_seconds() / 60.0)
        if position.risk_amount > 0:
            r_multiple = net_pnl / position.risk_amount
        else:
            r_multiple = 0.0

        record = TradeRecord(
            side=position.side,
            entry_timestamp=position.entry_time,
            entry_price=position.entry_price,
            exit_timestamp=timestamp,
            exit_price=exit_price,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            trade_duration_minutes=trade_duration_minutes,
            exit_reason=exit_reason,
        )

        self.trade_log.append(record)
        self.open_position = None
        return record

    def mark_to_market_equity(self, close_price: float) -> float:
        equity = self.capital
        if self.open_position is not None:
            if self.open_position.side == "LONG":
                equity += self.open_position.quantity * float(close_price)
            else:
                equity -= self.open_position.quantity * float(close_price)
        return equity

    def trade_log_dataframe(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame(
                columns=[
                    "side",
                    "entry_timestamp",
                    "entry_price",
                    "exit_timestamp",
                    "exit_price",
                    "quantity",
                    "gross_pnl",
                    "net_pnl",
                    "r_multiple",
                    "trade_duration_minutes",
                    "exit_reason",
                ]
            )

        return pd.DataFrame([t.__dict__ for t in self.trade_log])
