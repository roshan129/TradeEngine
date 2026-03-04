# TradeEngine

TradeEngine is a Python backend for market-data-driven trading workflows.

Current implementation includes:
- Sprint 1: Upstox historical + intraday candle integration
- Sprint 2: deterministic data validation and cleaning pipeline
- Sprint 3: deterministic feature engineering pipeline for indicators and context features

Trading strategies, order execution, position/risk management, and DB persistence are still out of scope.

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
    feature_validation_report.py
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
