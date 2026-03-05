from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Return maximum drawdown in percentage terms."""
    if equity_curve.empty:
        return 0.0

    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(abs(drawdown.min()) * 100.0)


def compute_sharpe_ratio(equity_curve: pd.Series) -> float:
    """Compute basic (non-annualized) Sharpe using per-candle returns."""
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return 0.0

    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0

    return float((returns.mean() / std) * math.sqrt(len(returns)))


def compute_performance_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute deterministic backtest summary metrics."""
    if equity_df.empty:
        return {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "expectancy": 0.0,
        }

    initial_equity = float(equity_df["equity"].iloc[0])
    final_equity = float(equity_df["equity"].iloc[-1])
    total_return_pct = ((final_equity / initial_equity) - 1.0) * 100.0

    if trades_df.empty:
        win_rate = 0.0
        average_win = 0.0
        average_loss = 0.0
        profit_factor = 0.0
        expectancy = 0.0
    else:
        net_pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").dropna()
        wins = net_pnl[net_pnl > 0]
        losses = net_pnl[net_pnl < 0]

        win_rate = float((len(wins) / len(net_pnl)) * 100.0) if len(net_pnl) > 0 else 0.0
        average_win = float(wins.mean()) if not wins.empty else 0.0
        average_loss = float(losses.mean()) if not losses.empty else 0.0

        gross_wins = float(wins.sum()) if not wins.empty else 0.0
        gross_losses = float(abs(losses.sum())) if not losses.empty else 0.0
        if gross_losses == 0.0:
            profit_factor = float("inf") if gross_wins > 0 else 0.0
        else:
            profit_factor = gross_wins / gross_losses

        expectancy = float(net_pnl.mean()) if not net_pnl.empty else 0.0

    max_drawdown_pct = compute_max_drawdown(equity_df["equity"])
    sharpe_ratio = compute_sharpe_ratio(equity_df["equity"])

    metrics = {
        "total_return_pct": float(total_return_pct),
        "win_rate": float(win_rate),
        "average_win": float(average_win),
        "average_loss": float(average_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe_ratio": float(sharpe_ratio),
        "expectancy": float(expectancy),
    }

    for key, value in metrics.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            # Keep profit_factor=inf when no losses; normalize all other non-finite values.
            if key == "profit_factor" and value == float("inf"):
                continue
            metrics[key] = 0.0

    return metrics
