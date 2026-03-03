# TradeEngine

TradeEngine is a Python backend for market-data-driven trading workflows.

Current implementation includes Sprint 1 for Upstox historical candles:
- Upstox config validation (fail-fast)
- OAuth/token-based authentication module
- Historical candle API client with retries and error handling
- Candle normalization into typed internal model
- Service methods for last 500 five-minute candles
- CLI execution flow
- FastAPI health endpoint

Trading strategies, order execution, and DB persistence are intentionally out of scope right now.

## Implemented Scope (Sprint 1)

- Historical source: Upstox
- Interval: `5minute`
- Target output: latest `500` normalized candles
- Timezone normalization: IST (`Asia/Kolkata`)
- Modes:
  - Market-hours aware: returns empty outside market hours
  - Ignore-market-hours: fetches regardless of market status

Sprint story reference:
- `SPRINT_1_UPSTOX_HISTORICAL_DATA.md`

## Project Layout

```text
TradeEngine/
  src/tradeengine/
    api/
      health.py
    auth/
      upstox_auth.py
    market_data/
      models.py
      service.py
      upstox_client.py
    utils/
      logger.py
    config.py
    main.py
  tests/
    test_config.py
    test_health.py
    test_market_data_models.py
    test_market_data_service.py
  .env.example
  requirements.txt
  pyproject.toml
```

## Prerequisites

- Python `3.11+`
- Upstox app credentials
- Upstox access token (sandbox or live, based on your usage)

## Setup

1. Create and activate virtual environment:
   - `python3.11 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `python -m pip install '.[dev]'`
3. Create env file:
   - `cp .env.example .env`
4. Fill required env vars in `.env`:
   - `UPSTOX_API_KEY`
   - `UPSTOX_API_SECRET`
   - `UPSTOX_REDIRECT_URI`
   - `UPSTOX_INSTRUMENT_KEY`
   - one of:
     - `UPSTOX_ACCESS_TOKEN` (recommended for sandbox flow)
     - or `UPSTOX_AUTH_CODE` (OAuth code flow)

## Run Historical Candle Flow

Market-hours aware flow:
- `PYTHONPATH=src .venv/bin/python -m tradeengine.main`

Ignore market-hours flow:
- `PYTHONPATH=src .venv/bin/python -m tradeengine.main --ignore-market-hours`

Optional CLI args:
- `--auth-code "<CODE>"`
- `--instrument-key "<INSTRUMENT_KEY>"`

## Run API Server

- `PYTHONPATH=src .venv/bin/python -m uvicorn tradeengine.main:app --reload --host 0.0.0.0 --port 8000`

Available endpoint:
- `GET /health` returns `{"status":"ok"}`

## Run Tests

- `PYTHONPATH=src .venv/bin/python -m pytest -q`
