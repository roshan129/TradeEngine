# TradeEngine

TradeEngine is a Python backend foundation for a stock trading bot platform.

This repository currently includes only initial project setup:
- FastAPI application scaffold
- Environment-based configuration
- Dependency and tooling setup
- Basic health endpoint
- Test and lint configuration

No trading logic is included yet.

## Project Layout

```text
TradeEngine/
  src/tradeengine/
    api/
      health.py
    config.py
    main.py
  tests/
    test_health.py
  .env.example
  .gitignore
  pyproject.toml
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -e .[dev]`
3. Copy env template:
   - `cp .env.example .env`
4. Run the API:
   - `uvicorn tradeengine.main:app --reload --host 0.0.0.0 --port 8000`

## Available Endpoints

- `GET /health` returns service health status.

