# TradeEngine

TradeEngine is a Python backend for market-data-driven trading workflows.

Current implementation includes:
- Sprint 1: Upstox historical + intraday candle integration
- Sprint 2: deterministic data validation and cleaning pipeline
- Sprint 3: deterministic feature engineering pipeline for indicators and context features
- Sprint 4: deterministic, event-driven backtesting engine
- Sprint 5: deterministic labeling engine and ML dataset builder
- Sprint 6: ML model training, evaluation, registry, prediction, and ML-driven backtests

Live order execution and DB persistence are still out of scope.

## Sprint 4: Backtesting Engine
- Stateless strategy contract (`BUY` / `SELL` / `SHORT` / `COVER` / `HOLD`)
- Multiple strategies:
  - `ema_rsi` (trend + momentum baseline)
  - `vwap_rsi_reversion` (mean reversion baseline)
  - `first_five_minute_momentum` (first 5-minute candle breakout using 1-minute execution)
  - `first_five_minute_fake_breakout` (fade failed breakouts of the first 5-minute candle)
  - `inside_bar_breakout` (current research default uses mother-candle range, full-day session, 1.5R, min mother range 0.35%, max 2 trades/day, stop after first winning trade)
  - `random_open_direction` (deterministic random long/short entry on a chosen intraday candle)
  - `one_minute_vwap_ema9_scalp` (VWAP+EMA9 pullback scalp with volume confirmation)
  - `support_resistance_reversal` (intraday swing-based support/resistance reversals)
  - supports multi-timeframe mode via 5-minute structure projected into 1-minute execution rows
- Candle-by-candle event-driven backtest loop (no vectorized leakage)
- Single-position portfolio model with dynamic ATR-based position sizing
- Stop loss, strategy exits, and end-of-day exits
- Brokerage and slippage cost simulation
- Structured trade log and equity curve generation
- Performance metrics:
  - return, win rate, avg win/loss, profit factor
  - gross wins/losses (before and after costs)
  - breakeven win rate
  - max drawdown, Sharpe, expectancy

Main classes:
- `tradeengine.core.backtester.Backtester`
- `tradeengine.core.portfolio.Portfolio`
- `tradeengine.core.strategy.BaselineEmaRsiStrategy`
- `tradeengine.core.strategy.VwapRsiMeanReversionStrategy`
- `tradeengine.core.metrics.compute_performance_metrics`

## Sprint 5: Labeling Engine & ML Dataset Builder
- Future-outcome labeling with leakage-safe forward shifts (`shift(-horizon)`)
- Fixed-threshold labels (`BUY` / `SELL` / `HOLD`)
- Multi-horizon return columns (`future_return_5`, `future_return_10`, `future_return_20`)
- Volatility-adjusted labels using ATR multiples
- ML dataset builder with strict schema + validation:
  - no NaN
  - no infinite numeric values
  - sorted timestamps
  - unique timestamps
- Class-distribution stats (`BUY`, `SELL`, `HOLD`) for imbalance checks

Main classes:
- `tradeengine.ml.labeling.LabelGenerator`
- `tradeengine.ml.dataset_builder.DatasetBuilder`

## Sprint 6: ML Training & Inference
- Feature config to prevent drift (`tradeengine.ml.models.feature_config`)
- XGBoost multi-class training with time-based split
- Class-balanced sample weights
- Evaluation metrics (accuracy, macro precision/recall/F1, confusion matrix, feature importance)
- Model registry (artifact + metadata JSON)
- Predictor with probability outputs
- CLI scripts: training, evaluation, prediction, walk-forward, OOS validation, OOS tuning
- ML-driven backtesting strategy (`ml_signal`) with entry time gate (default 09:20–10:20)

Main classes:
- `tradeengine.ml.models.trainer.ModelTrainer`
- `tradeengine.ml.models.predictor.ModelPredictor`
- `tradeengine.ml.models.registry.ModelRegistry`
- `tradeengine.ml.models.evaluation.evaluate_classification`

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
- Time/regime: `minute_of_day`, `minutes_since_open`, `session_progress`
- Opening behavior: `gap_percent`, `distance_from_open`, `distance_from_previous_close`
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
    ml/
      labeling.py
      dataset_builder.py
      models/
        trainer.py
        predictor.py
        registry.py
        evaluation.py
        feature_config.py
    config.py
    main.py
  scripts/
    dev.sh
    export_features_csv.py
    export_features_history.py
    export_features_1m.py
    feature_validation_report.py
    run_backtest.py
    build_ml_dataset.py
    train_model.py
    evaluate_model.py
    predict_from_model.py
    walk_forward_evaluate.py
    oos_validate_ml.py
    oos_tune_ml.py
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
- `data/market_data/features/feature_validation_output.csv`

## Export Multi-Month Feature History

Fetch historical 5-minute candles in date chunks, stitch deterministically, clean, and compute features:

- `PYTHONPATH=src .venv/bin/python scripts/export_features_history.py --from-date 2025-10-01 --to-date 2026-03-01 --chunk-days 28 --output data/market_data/features/feature_history_output.csv`

Optional:
- `--symbol NSE_EQ|INE848E01016` (or uses `UPSTOX_INSTRUMENT_KEY`)
- `--raw-output data/market_data/raw/raw_history_ohlcv.csv`
- `--pause-seconds 0.2`

Example:
- `PYTHONPATH=src .venv/bin/python scripts/export_features_history.py --from-date 2025-10-01 --to-date 2026-03-01 --chunk-days 28 --output data/market_data/features/feature_history_output.csv --raw-output data/market_data/raw/raw_history_ohlcv.csv`

## Current Inside-Bar Research Default

For `inside_bar_breakout`, the current research baseline is:

- mother candle breakout range
- full day `09:15-15:15`
- `1.5R`
- minimum mother candle range `0.35%`
- max `2` trades per day
- stop taking new entries after the first winning trade of the day
- volume and VWAP filters remain off unless enabled explicitly

## Current VWAP Continuation Hybrid Presets

The best candidates from the `2026-03-31` hybrid search are now saved as runnable presets in:

- `scripts/run_vwap_hybrid_preset.py`

Available presets:

- `sbin_best`
  - official SBIN preset now tracks the best 12-month winner
  - entry window `09:20-10:15`
  - breakout candle must close above setup high
  - chase cap `0.15%`
  - impulse filter `0.58%`
  - exit `rr` at `2R`
  - pullback depth `0.15% -> 0.35%`
  - default lookback `365` days
- `icici_best`
  - entry window `09:20-10:15`
  - breakout candle must close above setup high
  - chase cap `0.15%`
  - impulse filter `0.4%`
  - exit `trailing_low`
  - pullback depth `0.10% -> 0.25%`

These presets use the unlocked continuation logic:

- EMA filter removed
- VWAP slope filter removed
- candle-direction rule removed
- pullback defined as dip from recent high

The current `sbin_best` preset reproduces the best 12-month SBIN result found so far:

- return `+0.8147%`
- trades `9`
- profit factor `1.6348`
- max drawdown `0.8223%`

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_vwap_hybrid_preset.py \
  --preset sbin_best \
  --trades-output data/backtests/sbin_hybrid_preset_trades.csv \
  --equity-output data/backtests/sbin_hybrid_preset_equity.csv
```

## Export Multi-Month 1-Minute Feature History

Fetch historical 1-minute candles in date chunks, stitch deterministically, clean with 1-minute interval validation, compute features, and add `ema9`:

- `PYTHONPATH=src .venv/bin/python scripts/export_features_1m.py --from-date 2025-10-01 --to-date 2026-03-01 --chunk-days 28 --output data/market_data/features/feature_history_1m_output.csv`

Optional:
- `--symbol NSE_EQ|INE848E01016` (or uses `UPSTOX_INSTRUMENT_KEY`)
- `--raw-output data/market_data/raw/raw_history_1m_ohlcv.csv`
- `--pause-seconds 0.2`

## Validate Indicators Against Reference CSV

Use comparison report script:

```bash
PYTHONPATH=src .venv/bin/python scripts/feature_validation_report.py \
  --computed data/market_data/features/feature_validation_output.csv \
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

Run Sprint 5 labeling tests:
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_labeling.py`

Run backtest CLI:
- `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_validation_output.csv`

EMA+RSI strategy commands:
- Long-only:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy ema_rsi`
- Long+Short:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy ema_rsi --allow-shorts`
- Reversed signals:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy ema_rsi --reverse-signals`

VWAP+RSI reversion strategy commands:
- Long-only:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy vwap_rsi_reversion`
- Long+Short (`SHORT` when `close > vwap` and `rsi > 65`):
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy vwap_rsi_reversion --allow-shorts`
- Reversed signals:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_output.csv --strategy vwap_rsi_reversion --reverse-signals`

Random open-direction strategy commands:
- 5-minute random long/short at `09:15` with `1:1` RR:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_5m_sbin.csv --strategy random_open_direction --allow-shorts --random-entry-time 09:15 --random-rr-multiple 1.0 --random-seed 42`

First 5-minute candle momentum strategy commands:
- Breakout of the `09:15-09:20` candle on 1-minute data with `1:1` RR:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_sbin.csv --strategy first_five_minute_momentum --allow-shorts --first-candle-rr-multiple 1.0`
- Breakout of the `09:15-09:20` candle with fixed `0.25%` stop and `0.25%` target:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_sbin.csv --strategy first_five_minute_momentum --allow-shorts --first-candle-stop-loss-pct 0.0025 --first-candle-take-profit-pct 0.0025`

First 5-minute fake breakout strategy commands:
- Fade failed first-candle breakouts on 1-minute data with `1:1` RR:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_sbin.csv --strategy first_five_minute_fake_breakout --allow-shorts --fake-breakout-rr-multiple 1.0`
- Fade failed first-candle breakouts with fixed `0.25%` stop and `0.5%` target:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_sbin.csv --strategy first_five_minute_fake_breakout --allow-shorts --fake-breakout-stop-loss-pct 0.0025 --fake-breakout-take-profit-pct 0.005`

One-minute VWAP+EMA9 scalp strategy commands:
- Long-only:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_scalp`
- Long+Short:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_scalp --allow-shorts`
- Reversed signals:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_scalp --reverse-signals`
- ATR take-profit mode:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_scalp --scalp-tp-mode atr`

ICICI-focused one-minute strategy commands:
- Long+Short with stricter filters:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_icici --allow-shorts`
- Apply per-day entry cap:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_icici --allow-shorts --max-entries-per-day 6`
- ATR take-profit mode:
- `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy one_minute_vwap_ema9_icici --allow-shorts --scalp-tp-mode atr`

Support/resistance reversal strategy:
- Single-timeframe on the input CSV:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_1m_output.csv --strategy support_resistance_reversal --allow-shorts`
- Multi-timeframe: detect structure on 5-minute data, execute on 1-minute data:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_nifty50.csv --structure-input data/market_data/features/feature_history_36m_5m_nifty50.csv --strategy support_resistance_reversal --allow-shorts`

## Recent Breakout Research

SBI (`NSE_EQ|INE062A01020`) has been the primary single-stock research dataset for recent breakout testing.

- `support_resistance_reversal` was not viable on SBI, even in reversed or MTF variants.
- `first_five_minute_momentum` has been the strongest breakout-style baseline so far.
- Reversing the first-candle breakout did not create an edge after stop-loss handling was fixed.
- `first_five_minute_fake_breakout` underperformed the plain first-candle breakout on SBI.

Current least-bad tested setup on SBI:
- Strategy: `first_five_minute_momentum`
- Dataset: `data/market_data/features/feature_history_36m_1m_sbin.csv`
- Gap filter: `abs(gap_percent) <= 0.5`
- Breakout window: `09:20-09:30`
- Fixed stop loss: `0.25%`
- Fixed take profit: `0.25%`

Reference command:
- `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/market_data/features/feature_history_36m_1m_sbin.csv --strategy first_five_minute_momentum --allow-shorts --first-candle-stop-loss-pct 0.0025 --first-candle-take-profit-pct 0.0025 --first-candle-max-gap-percent 0.5 --first-candle-breakout-end 09:30`

ML-driven strategy (uses `prediction` column):
- `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/ml/predictions_12m_5m_t2_full.csv --strategy ml_signal --allow-shorts`
- Configure entry window:
  - `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py --input data/ml/predictions_12m_5m_t2_full.csv --strategy ml_signal --allow-shorts --ml-entry-start 09:20 --ml-entry-end 10:20`

Build ML dataset:
- `PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py --input data/market_data/features/feature_history_output.csv --output data/ml/ml_dataset.csv`

Optional ML flags:
- `--horizons 5,10,20`
- `--label-horizon 5`
- `--buy-threshold 0.003`
- `--sell-threshold -0.003`
- `--use-volatility-adjusted-labels`
- `--atr-multiplier 0.5`

Train ML model:
- `PYTHONPATH=src .venv/bin/python scripts/train_model.py --dataset data/ml/ml_dataset.csv --output models/model_v1.pkl --model-version v1`

Evaluate ML model:
- `PYTHONPATH=src .venv/bin/python scripts/evaluate_model.py --dataset data/ml/ml_dataset.csv --model models/model_v1.pkl`

Walk-forward evaluation:
- `PYTHONPATH=src .venv/bin/python scripts/walk_forward_evaluate.py --dataset data/ml/ml_dataset.csv --model models/model_v1.pkl`

Generate predictions with gating:
- `PYTHONPATH=src .venv/bin/python scripts/predict_from_model.py --model models/model_v1.pkl --input data/market_data/features/feature_validation_output.csv --output data/ml/predictions.csv --buy-threshold-proba 0.45 --sell-threshold-proba 0.55`

OOS validation (9m train / 3m test):
- `PYTHONPATH=src .venv/bin/python scripts/oos_validate_ml.py --features data/market_data/features/feature_history_12m_5m_v2.csv --model-output models/model_12m_5m_oos_9m3m_v2.pkl --output-dir oos_results_12m_5m_v2`

OOS tuning (threshold + risk + stop):
- `PYTHONPATH=src .venv/bin/python scripts/oos_tune_ml.py --features data/market_data/features/feature_history_12m_5m_v2.csv --output-dir oos_tune_results_12m_5m_v2 --model-output models/model_12m_5m_tuned_oos_v2.pkl`

## Set Up A Disciplined Research Loop

Create a research window, an untouched holdout window, and a markdown experiment log:

- `PYTHONPATH=src .venv/bin/python scripts/setup_research_workflow.py --input data/market_data/features/feature_history_36m_5m.csv --strategy-name inside_bar_breakout --research-months 9 --holdout-months 3 --output-dir research_inside_bar`

VWAP continuation now has a dedicated research scaffold with tuned liquid-name defaults:

- `PYTHONPATH=src .venv/bin/python scripts/setup_vwap_continuation_research.py --input data/market_data/features/feature_history_36m_5m_sbin.csv --strategy-name vwap_trend_continuation --research-months 9 --holdout-months 3 --output-dir research_vwap_trend_continuation_sbin`

Current tuned baseline for `vwap_trend_continuation`:

- long-only
- entry window `09:20-10:45`
- exit mode `vwap_break`
- minimum `5` candles above VWAP before setup
- minimum VWAP distance `0.15%`
- pullback size `0.15%`
- fixed stop cap `0.30%`
- max `1` trade/day
- stop after first winning trade

The script writes:

- `research_window.csv`
- `holdout_window.csv`
- `research_weekly_summary.csv`
- `research_monthly_summary.csv`
- `RESEARCH_WORKFLOW.md`
