from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.core.metrics import compute_performance_metrics


def test_metrics_match_known_values() -> None:
    trades = pd.DataFrame(
        {
            "net_pnl": [10.0, -5.0, 15.0],
        }
    )
    equity = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 105.0, 120.0],
        }
    )

    metrics = compute_performance_metrics(trades, equity)

    assert metrics["total_return_pct"] == pytest.approx(20.0)
    assert metrics["win_rate"] == pytest.approx((2 / 3) * 100.0)
    assert metrics["average_win"] == pytest.approx(12.5)
    assert metrics["average_loss"] == pytest.approx(-5.0)
    assert metrics["profit_factor"] == pytest.approx(5.0)
    assert metrics["max_drawdown_pct"] == pytest.approx((5 / 110) * 100.0)
    assert metrics["expectancy"] == pytest.approx((10.0 - 5.0 + 15.0) / 3.0)
