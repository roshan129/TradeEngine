# TradeEngine

TradeEngine is a Python backend for market-data-driven trading workflows.

Current implementation includes:
- Sprint 1: Upstox historical + intraday candle integration
- Sprint 2: deterministic data validation and cleaning pipeline
- Sprint 3: deterministic feature engineering pipeline for indicators and context features
- Sprint 4: deterministic, event-driven backtesting engine

Live order execution and DB persistence are still out of scope.

## Sprint 4: Backtesting Engine
- Stateless strategy contract (`BUY` / `SELL` / `HOLD`)
- Baseline EMA+RSI long-only strategy
- Candle-by-candle event-driven backtest loop (no vectorized leakage)
- Single-position portfolio model with dynamic ATR-based position sizing
- Stop loss, RSI exit, and end-of-day exits
- Brokerage and slippage cost simulation
- Structured trade log and equity curve generation
- Performance metrics: return, win rate, avg win/loss, profit factor, max drawdown, Sharpe, expectancy

Main classes:
- `tradeengine.core.backtester.Backtester`
- `tradeengine.core.portfolio.Portfolio`
- `tradeengine.core.strategy.BaselineEmaRsiStrategy`
- `tradeengine.core.metrics.compute_performance_metrics`

## Implemented Scope

### Sprint 1: Market Data Integration
- Upstox config validation (fail-fast)
- OAuth/token-based authentication module
- Historical + intraday candle API client with retry/backoff
- Candle normalization into typed internal model
- Service methods for latest 500 five-minute candles
- CLI flow and FastAPI health endpoint

### Sprint 2: Data Validation & Cleaning
- Canonical candle schema validation
- Timestamp normalization to IST
- Sorting and deduplication
- Missing interval detection with structured warning logs
- Numeric type casting with NaN checks
- Logical OHLCV validation
- Full clean output contract

Main class:
- `tradeengine.core.data_processor.MarketDataProcessor`

### Sprint 3: Feature Engineering
- Trend: `ema20`, `ema50`, `ema200`, `vwap`
- Momentum: `rsi`, `macd`, `macd_signal`, `macd_hist`, `roc`
- Volatility: `atr`, `bb_width`, `rolling_std`
- Structure/context: `dist_ema20`, `dist_vwap`, `higher_high`, `lower_low`, `rolling_volume_avg`
- Warmup handling (`dropna` after indicator generation)
- Full feature contract checks (no NaN, no duplicates, sorted timestamps, finite numerics)

Main class:
- `tradeengine.core.features.FeatureEngineer`

## Project Layout

```text
TradeEngine/
  src/tradeengine/
    api/
      health.py
    auth/
      upstox_auth.py
    core/
      data_processor.py
      features.py
      strategy.py
      portfolio.py
      backtester.py
      metrics.py
    market_data/
      models.py
      service.py
      upstox_client.py
    utils/
      logger.py
    config.py
    main.py
  scripts/
    dev.sh
    export_features_csv.py
    export_features_history.py
    feature_validation_report.py
    run_backtest.py
  tests/
    test_*.py
  .env.example
  pyproject.toml
  requirements.txt
```

## Prerequisites

- Python `3.11+`
- Upstox app credentials
- Upstox token (`UPSTOX_ACCESS_TOKEN`) and instrument key

## Setup

1. Create and activate venv:
   - `python3.11 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `python -m pip install '.[dev]'`
3. Create env file:
   - `cp .env.example .env`
4. Fill `.env` at minimum:
   - `UPSTOX_API_KEY`
   - `UPSTOX_API_SECRET`
   - `UPSTOX_REDIRECT_URI`
   - `UPSTOX_ACCESS_TOKEN`
   - `UPSTOX_INSTRUMENT_KEY`

## Run Market Data Flow

Market-hours aware:
- `PYTHONPATH=src .venv/bin/python -m tradeengine.main`

Ignore market-hours gate:
- `PYTHONPATH=src .venv/bin/python -m tradeengine.main --ignore-market-hours`

## Export Feature CSV

One-command CSV export (latest timestamp first by default):
- `PYTHONPATH=src .venv/bin/python scripts/export_features_csv.py`

Optional flags:
- `--output my_features.csv`
- `--oldest-first`
- `--respect-market-hours`

Default output file:
- `feature_validation_output.csv`

## Export Multi-Month Feature History

Fetch historical 5-minute candles in date chunks, stitch deterministically, clean, and compute features:

- `PYTHONPATH=src .venv/bin/python scripts/export_features_history.py --from-date 2025-10-01 --to-date 2026-03-01 --chunk-days 28 --output feature_history_output.csv`

Optional:
- `--symbol NSE_EQ|INE848E01016` (or uses `UPSTOX_INSTRUMENT_KEY`)
- `--raw-output raw_history_ohlcv.csv`
- `--pause-seconds 0.2`

## Validate Indicators Against Reference CSV

Use comparison report script:

```bash
PYTHONPATH=src .venv/bin/python scripts/feature_validation_report.py \
  --computed feature_validation_output.csv \
  --reference tradingview_export.csv \
  --tolerance 0.05
```

If column names differ, map them:
- `--metric ema20:"EMA 20"` (repeat for each metric)

## Run API Server

- `PYTHONPATH=src .venv/bin/python -m uvicorn tradeengine.main:app --reload --host 0.0.0.0 --port 8000`

Endpoint:
- `GET /health` -> `{"status":"ok"}`

## Run Tests

All tests:
- `PYTHONPATH=src .venv/bin/python -m pytest -q`

Sprint 3 feature tests only:
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_feature_engineer_story1.py tests/test_feature_engineer_trend.py tests/test_feature_engineer_momentum.py tests/test_feature_engineer_volatility.py tests/test_feature_engineer_structure.py tests/test_feature_engineer_warmup.py tests/test_feature_engineer_pipeline.py`

Run backtest tests:
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_strategy_backtesting.py tests/test_portfolio.py tests/test_backtester.py tests/test_metrics.py`

Run backtest CLI:
- `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv`
- Select strategy:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy ema_rsi`
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy vwap_rsi_reversion`
- Enable short mode:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy ema_rsi --allow-shorts`
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy vwap_rsi_reversion --allow-shorts`
- Reverse long/short signals:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy ema_rsi --reverse-signals`
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input feature_validation_output.csv --strategy vwap_rsi_reversion --reverse-signals`
