from __future__ import annotations

import pandas as pd

from tradeengine.core.portfolio import CostModel, Portfolio


def test_portfolio_position_lifecycle_single_position() -> None:
    portfolio = Portfolio(initial_capital=10_000.0, cost_model=CostModel(0.0, 0.0, 0.0))

    entered = portfolio.enter_position(
        side="LONG",
        timestamp=pd.Timestamp("2026-01-01T09:15:00+05:30"),
        candle_close=100.0,
        stop_loss=98.0,
        quantity=10,
        entry_index=0,
        risk_amount=100.0,
    )

    assert entered
    assert portfolio.has_open_position
    assert portfolio.capital == 9_000.0

    second_enter = portfolio.enter_position(
        side="LONG",
        timestamp=pd.Timestamp("2026-01-01T09:20:00+05:30"),
        candle_close=101.0,
        stop_loss=99.0,
        quantity=5,
        entry_index=1,
        risk_amount=50.0,
    )
    assert not second_enter

    trade = portfolio.exit_position(
        timestamp=pd.Timestamp("2026-01-01T09:25:00+05:30"),
        candle_price=110.0,
        exit_reason="TEST_EXIT",
    )

    assert not portfolio.has_open_position
    assert portfolio.capital == 10_100.0
    assert trade.gross_pnl == 100.0
    assert trade.net_pnl == 100.0
    assert len(portfolio.trade_log) == 1
