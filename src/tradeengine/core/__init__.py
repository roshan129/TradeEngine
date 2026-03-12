"""Core data processing, feature engineering, and backtesting components."""

from tradeengine.core.backtester import (
    BacktestConfig,
    Backtester,
    BacktestError,
    BacktestResult,
)
from tradeengine.core.data_processor import DataSchemaError, MarketDataProcessor
from tradeengine.core.features import FeatureEngineer, FeatureEngineeringError
from tradeengine.core.metrics import (
    compute_max_drawdown,
    compute_performance_metrics,
    compute_sharpe_ratio,
)
from tradeengine.core.portfolio import CostModel, Portfolio, PortfolioError, Position, TradeRecord
from tradeengine.core.strategy import (
    BaselineEmaRsiStrategy,
    OpeningRangeBreakoutStrategy,
    OneMinuteVwapEma9IciciFocusedStrategy,
    OneMinuteVwapEma9ScalpStrategy,
    Signal,
    Strategy,
    StrategyContext,
    VwapRsiMeanReversionStrategy,
)

__all__ = [
    "MarketDataProcessor",
    "DataSchemaError",
    "FeatureEngineer",
    "FeatureEngineeringError",
    "Signal",
    "Strategy",
    "StrategyContext",
    "BaselineEmaRsiStrategy",
    "OpeningRangeBreakoutStrategy",
    "VwapRsiMeanReversionStrategy",
    "OneMinuteVwapEma9ScalpStrategy",
    "OneMinuteVwapEma9IciciFocusedStrategy",
    "CostModel",
    "Position",
    "TradeRecord",
    "Portfolio",
    "PortfolioError",
    "BacktestConfig",
    "BacktestError",
    "BacktestResult",
    "Backtester",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_performance_metrics",
]
