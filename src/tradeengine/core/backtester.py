from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradeengine.core.metrics import compute_performance_metrics
from tradeengine.core.portfolio import CostModel, Portfolio
from tradeengine.core.strategy import Strategy, StrategyContext


class BacktestError(ValueError):
    """Raised when backtester preconditions are invalid."""


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    stop_atr_multiple: float = 1.0
    slippage_pct: float = 0.0005
    brokerage_fixed: float = 20.0
    brokerage_pct: float = 0.0003
    force_end_of_day_exit: bool = True
    allow_shorts: bool = False


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


class Backtester:
    BASE_REQUIRED_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    )

    def __init__(self, strategy: Strategy, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def _required_columns(self) -> tuple[str, ...]:
        strategy_columns = getattr(self.strategy, "required_columns", ())
        return self.BASE_REQUIRED_COLUMNS + tuple(strategy_columns)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = self._required_columns()
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise BacktestError(f"Missing required columns for backtest: {', '.join(missing)}")

        clean = df.copy(deep=True)
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
        if clean["timestamp"].isna().any():
            raise BacktestError("Backtest dataframe contains invalid timestamps")

        clean = clean.sort_values("timestamp", ascending=True).reset_index(drop=True)
        if clean["timestamp"].duplicated().any():
            raise BacktestError("Backtest dataframe contains duplicate timestamps")

        numeric_cols = [c for c in required_columns if c != "timestamp"]
        for col in numeric_cols:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        if clean[numeric_cols].isna().any().any():
            raise BacktestError("Backtest dataframe contains NaN in required numeric columns")

        return clean

    def _is_end_of_day(self, df: pd.DataFrame, idx: int) -> bool:
        if idx >= len(df) - 1:
            return True
        return df.loc[idx, "timestamp"].date() != df.loc[idx + 1, "timestamp"].date()

    def _compute_quantity(
        self,
        capital: float,
        close_price: float,
        stop_distance: float,
        side: str,
    ) -> tuple[int, float]:
        if stop_distance <= 0 or close_price <= 0 or capital <= 0:
            return 0, 0.0

        risk_amount = capital * self.config.risk_per_trade
        qty_by_risk = int(np.floor(risk_amount / stop_distance))

        if side == "LONG":
            entry_fill = close_price * (1.0 + self.config.slippage_pct)
        else:
            entry_fill = close_price * (1.0 - self.config.slippage_pct)
        qty_by_capital = int(np.floor(capital / entry_fill))
        quantity = max(0, min(qty_by_risk, qty_by_capital))

        return quantity, risk_amount

    def run(self, df: pd.DataFrame) -> BacktestResult:
        candles = self._prepare_dataframe(df)
        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            cost_model=CostModel(
                brokerage_fixed=self.config.brokerage_fixed,
                brokerage_pct=self.config.brokerage_pct,
                slippage_pct=self.config.slippage_pct,
            ),
        )

        equity_rows: list[dict[str, object]] = []

        for i in range(len(candles)):
            row = candles.iloc[i]
            timestamp = pd.Timestamp(row["timestamp"])
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            is_eod = self._is_end_of_day(candles, i)
            exited_this_candle = False

            if portfolio.open_position is not None:
                position = portfolio.open_position
                stop_hit = (
                    low <= position.stop_loss
                    if position.side == "LONG"
                    else high >= position.stop_loss
                )
                if i > position.entry_index and stop_hit:
                    portfolio.exit_position(
                        timestamp=timestamp,
                        candle_price=position.stop_loss,
                        exit_reason="STOP_LOSS",
                        use_stop_fill=True,
                    )
                    exited_this_candle = True

            context = StrategyContext(
                in_position=portfolio.has_open_position,
                available_capital=portfolio.capital,
                is_end_of_day=is_eod,
                position_side=portfolio.open_position.side if portfolio.open_position else None,
            )
            signal = self.strategy.generate_signal(row, context)

            if portfolio.open_position is None:
                if exited_this_candle:
                    equity_rows.append(
                        {
                            "timestamp": timestamp,
                            "capital": float(portfolio.capital),
                            "equity": float(portfolio.mark_to_market_equity(close)),
                            "in_position": portfolio.has_open_position,
                        }
                    )
                    continue

                if signal == "BUY":
                    stop_loss = self.strategy.entry_stop_loss(
                        row=row,
                        signal=signal,
                        stop_atr_multiple=self.config.stop_atr_multiple,
                    )
                    if stop_loss is not None:
                        stop_distance = abs(close - stop_loss)
                        quantity, risk_amount = self._compute_quantity(
                            capital=portfolio.capital,
                            close_price=close,
                            stop_distance=stop_distance,
                            side="LONG",
                        )
                        if quantity > 0:
                            portfolio.enter_position(
                                side="LONG",
                                timestamp=timestamp,
                                candle_close=close,
                                stop_loss=stop_loss,
                                quantity=quantity,
                                entry_index=i,
                                risk_amount=risk_amount,
                            )
                elif signal == "SHORT" and self.config.allow_shorts:
                    stop_loss = self.strategy.entry_stop_loss(
                        row=row,
                        signal=signal,
                        stop_atr_multiple=self.config.stop_atr_multiple,
                    )
                    if stop_loss is not None:
                        stop_distance = abs(close - stop_loss)
                        quantity, risk_amount = self._compute_quantity(
                            capital=portfolio.capital,
                            close_price=close,
                            stop_distance=stop_distance,
                            side="SHORT",
                        )
                        if quantity > 0:
                            portfolio.enter_position(
                                side="SHORT",
                                timestamp=timestamp,
                                candle_close=close,
                                stop_loss=stop_loss,
                                quantity=quantity,
                                entry_index=i,
                                risk_amount=risk_amount,
                            )
            else:
                if signal in {"SELL", "COVER"}:
                    portfolio.exit_position(
                        timestamp=timestamp,
                        candle_price=close,
                        exit_reason=(
                            "STRATEGY_EXIT_LONG" if signal == "SELL" else "STRATEGY_EXIT_SHORT"
                        ),
                        use_stop_fill=False,
                    )
                elif self.config.force_end_of_day_exit and is_eod:
                    portfolio.exit_position(
                        timestamp=timestamp,
                        candle_price=close,
                        exit_reason="END_OF_DAY",
                        use_stop_fill=False,
                    )

            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "capital": float(portfolio.capital),
                    "equity": float(portfolio.mark_to_market_equity(close)),
                    "in_position": portfolio.has_open_position,
                }
            )

        if portfolio.open_position is not None:
            last_row = candles.iloc[-1]
            portfolio.exit_position(
                timestamp=pd.Timestamp(last_row["timestamp"]),
                candle_price=float(last_row["close"]),
                exit_reason="BACKTEST_END",
                use_stop_fill=False,
            )
            equity_rows[-1]["capital"] = float(portfolio.capital)
            equity_rows[-1]["equity"] = float(portfolio.capital)
            equity_rows[-1]["in_position"] = False

        trades_df = portfolio.trade_log_dataframe()
        equity_df = pd.DataFrame(equity_rows)
        metrics = compute_performance_metrics(trades_df, equity_df)

        return BacktestResult(trades=trades_df, equity_curve=equity_df, metrics=metrics)
