# TradeEngine Scripts Reference

This file lists the scripts in `scripts/` with example commands and brief explanations.

Notes:
- If the package is not installed in editable mode, prefix commands with `PYTHONPATH=src`.
- For Upstox fetch/export scripts, set `UPSTOX_ACCESS_TOKEN` and `UPSTOX_INSTRUMENT_KEY` in `.env`
  (or pass `--symbol` where supported).

**build_ml_dataset.py**
```bash
# Build an ML dataset from a feature history CSV (labels + horizons).
PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py \
  --input feature_history_36m_5m.csv \
  --output ml_dataset_36m_5m.csv \
  --horizons 5,10,20 \
  --label-horizon 5 \
  --buy-threshold 0.003 \
  --sell-threshold -0.003
```

**train_model.py**
```bash
# Train an ML classifier from an ML dataset CSV and save a model artifact.
PYTHONPATH=src .venv/bin/python scripts/train_model.py \
  --dataset ml_dataset_36m_5m.csv \
  --output models/model_36m_5m.pkl \
  --model-version orb_36m_5m_v1 \
  --allow-missing-features
```

**evaluate_model.py**
```bash
# Evaluate a saved model on a dataset (single split or walk-forward).
PYTHONPATH=src .venv/bin/python scripts/evaluate_model.py \
  --dataset ml_dataset_36m_5m.csv \
  --model models/model_36m_5m.pkl
```

**walk_forward_evaluate.py**
```bash
# Walk-forward evaluation of a saved model with optional session filtering.
PYTHONPATH=src .venv/bin/python scripts/walk_forward_evaluate.py \
  --dataset ml_dataset_36m_5m.csv \
  --model models/model_36m_5m.pkl \
  --initial-train-fraction 0.6 \
  --test-fraction 0.2 \
  --step-fraction 0.1 \
  --session-start 09:20 \
  --session-end 10:20
```

**predict_from_model.py**
```bash
# Generate predictions and probabilities for a features CSV using a saved model.
PYTHONPATH=src .venv/bin/python scripts/predict_from_model.py \
  --model models/model_36m_5m.pkl \
  --input feature_history_36m_5m.csv \
  --output predictions.csv \
  --buy-threshold-proba 0.65 \
  --sell-threshold-proba 0.65
```

**run_backtest.py**
```bash
# Run a candle-by-candle backtest (example: Opening Range Breakout with ML filter).
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_5m.csv \
  --strategy opening_range_breakout \
  --model models/model_36m_5m.pkl \
  --opening-start 09:15 \
  --opening-end 09:30 \
  --orb-sl-pct 0.004 \
  --orb-tp-pct 0.006 \
  --orb-prob-threshold 0.65 \
  --max-entries-per-day 3
```

```bash
# Run Inside Bar Breakout using the current research defaults.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_15m_output.csv \
  --strategy inside_bar_breakout \
  --allow-shorts
```

Current default research config for `inside_bar_breakout`:
- mother candle breakout range
- full day `09:15-15:15`
- `1.5R`
- minimum mother candle range `0.35%`
- max `2` trades per day
- stop after first winning trade of the day
- volume and VWAP filters disabled unless explicitly enabled

```bash
# Example override: use inside-bar range instead of mother-candle range.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_15m_output.csv \
  --strategy inside_bar_breakout \
  --allow-shorts \
  --inside-use-inside-range
```

```bash
# Run Support/Resistance Reversal on 1-minute data with volume + VWAP filters.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_1m_output.csv \
  --strategy support_resistance_reversal \
  --sr-entry-start 09:20 \
  --sr-entry-end 14:30 \
  --sr-risk-reward 1.5 \
  --max-entries-per-day 10
```

```bash
# Run deterministic random long/short entries on the 09:15 5-minute candle with 1:1 RR.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_5m_sbin.csv \
  --strategy random_open_direction \
  --allow-shorts \
  --random-entry-time 09:15 \
  --random-rr-multiple 1.0 \
  --random-seed 42
```

```bash
# Run the first 5-minute candle momentum breakout strategy on 1-minute data.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_1m_sbin.csv \
  --strategy first_five_minute_momentum \
  --allow-shorts \
  --first-candle-rr-multiple 1.0
```

```bash
# Run the first 5-minute fake-breakout strategy on 1-minute data.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_1m_sbin.csv \
  --strategy first_five_minute_fake_breakout \
  --allow-shorts \
  --fake-breakout-rr-multiple 1.0
```

```bash
# Compare the current least-bad SBI first-candle breakout setup:
# fixed 0.25% stop, fixed 0.25% target, small-gap days only, breakout by 09:30.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_1m_sbin.csv \
  --strategy first_five_minute_momentum \
  --allow-shorts \
  --first-candle-stop-loss-pct 0.0025 \
  --first-candle-take-profit-pct 0.0025 \
  --first-candle-max-gap-percent 0.5 \
  --first-candle-breakout-end 09:30
```

```bash
# Compare first-candle breakout vs fake-breakout with fixed stop/target.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_1m_sbin.csv \
  --strategy first_five_minute_fake_breakout \
  --allow-shorts \
  --fake-breakout-stop-loss-pct 0.0025 \
  --fake-breakout-take-profit-pct 0.005
```

```bash
# Run multi-timeframe Support/Resistance Reversal:
# 5-minute candles build structure levels, 1-minute candles execute trades.
PYTHONPATH=src .venv/bin/python scripts/run_backtest.py \
  --input feature_history_36m_1m_nifty50.csv \
  --structure-input feature_history_36m_5m_nifty50.csv \
  --strategy support_resistance_reversal \
  --allow-shorts \
  --sr-risk-reward 1.5 \
  --max-entries-per-day 10
```

**walk_forward_orb_ml.py**
```bash
# Monthly walk-forward: train ML each fold, then backtest ORB on next month(s).
PYTHONPATH=src .venv/bin/python scripts/walk_forward_orb_ml.py \
  --features feature_history_36m_5m.csv \
  --output-dir walk_forward_orb_results \
  --initial-train-months 12 \
  --test-months 2 \
  --step-months 2 \
  --opening-start 09:15 \
  --opening-end 09:30 \
  --orb-sl-pct 0.004 \
  --orb-tp-pct 0.006 \
  --orb-prob-threshold 0.65
```

**oos_validate_ml.py**
```bash
# Train on earlier months and evaluate on the most recent test window (OOS).
PYTHONPATH=src .venv/bin/python scripts/oos_validate_ml.py \
  --features feature_history_12m_5m.csv \
  --output-dir oos_results \
  --model-output models/model_oos.pkl \
  --test-months 3
```

**oos_tune_ml.py**
```bash
# Tune ML probability thresholds and risk/stop grids on train window, evaluate on OOS.
PYTHONPATH=src .venv/bin/python scripts/oos_tune_ml.py \
  --features feature_history_12m_5m.csv \
  --output-dir oos_tune_results \
  --model-output models/model_oos_tuned.pkl \
  --test-months 3
```

**sweep_icici_strategy.py**
```bash
# Sweep ICICI 1-minute strategy parameters and write ranked results to CSV.
PYTHONPATH=src .venv/bin/python scripts/sweep_icici_strategy.py \
  --input feature_history_1m_output.csv \
  --output icici_sweep_results.csv \
  --max-entries 4,6,8 \
  --volume-multipliers 1.6,1.8,2.0 \
  --risk-rewards 1.5,1.8,2.0 \
  --session-ends 14:00,14:30,14:45
```

**feature_validation_report.py**
```bash
# Compare computed feature CSV against a reference export and print error stats.
PYTHONPATH=src .venv/bin/python scripts/feature_validation_report.py \
  --computed feature_history_output.csv \
  --reference reference_features.csv \
  --tolerance 0.05
```

**export_features_history.py**
```bash
# Fetch and stitch historical 5-minute candles from Upstox, then generate features CSV.
PYTHONPATH=src .venv/bin/python scripts/export_features_history.py \
  --from-date 2023-03-01 \
  --to-date 2026-03-01 \
  --output feature_history_36m_5m.csv \
  --raw-output raw_history_36m_5m.csv
```

**export_features_1m.py**
```bash
# Fetch and stitch historical 1-minute candles from Upstox, then generate features CSV.
PYTHONPATH=src .venv/bin/python scripts/export_features_1m.py \
  --from-date 2025-01-01 \
  --to-date 2025-12-31 \
  --output feature_history_1m_output.csv \
  --raw-output raw_history_1m.csv
```

**export_features_csv.py**
```bash
# Export engineered features from the most recent 5-minute candles (Upstox).
PYTHONPATH=src .venv/bin/python scripts/export_features_csv.py \
  --output feature_validation_output.csv \
  --oldest-first
```

**dev.sh**
```bash
# Run the local API server (Uvicorn) for TradeEngine.
bash scripts/dev.sh
```
