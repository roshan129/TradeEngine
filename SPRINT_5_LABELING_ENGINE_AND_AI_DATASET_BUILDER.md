# Sprint 5 - Labeling Engine & AI Dataset Builder

## Sprint Goal
Create a deterministic labeling system that converts the engineered feature dataset into an ML-ready dataset by attaching future-outcome labels.

Final output:
- `ml_dataset.csv`

Each row should contain:
- features at time `t`
- label describing what happened after time `t`

## Why Labeling Matters
Current dataset includes indicators/features/market state but not outcome quality.
Labeling answers:
- what happened after this candle?
- did price move up/down?
- did it move enough to trade?

## Module Structure
Create:

```text
src/tradeengine/ml/
  labeling.py
  dataset_builder.py
```

## Story 1 - Labeling Engine
File:
- `src/tradeengine/ml/labeling.py`

Class:
- `LabelGenerator`

Core API:
- `generate_labels(df, horizon=5, buy_threshold=0.003, sell_threshold=-0.003)`

Labeling logic:
- `future_close = close.shift(-horizon)`
- `future_return = (future_close - close) / close`
- `BUY` if `future_return > buy_threshold`
- `SELL` if `future_return < sell_threshold`
- else `HOLD`
- drop rows where future data is unavailable

Output columns:
- `future_close`
- `future_return`
- `label`

## Story 2 - Multi-Horizon Labels
Generate multiple future-return columns:
- `future_return_5`
- `future_return_10`
- `future_return_20`

## Story 3 - Volatility-Adjusted Labels
Add ATR-scaled thresholds (optional mode):
- `BUY` if future move > `k * ATR`
- `SELL` if future move < `-k * ATR`

## Story 4 - Dataset Builder
File:
- `src/tradeengine/ml/dataset_builder.py`

Class:
- `DatasetBuilder`

API:
- `build_dataset(df)`

Pipeline:
1. load feature dataset
2. generate labels
3. remove leakage columns
4. return ML dataset

Leakage columns to remove from final model features:
- `future_close`
- `future_return`

## Story 5 - Dataset Export Script
Create:
- `scripts/build_ml_dataset.py`

CLI example:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py \
  --input feature_history_output.csv \
  --output ml_dataset.csv
```

## Story 6 - Dataset Validation
Validation contract:
- no NaN
- no infinite values
- sorted timestamps
- no duplicate timestamps

## Story 7 - Dataset Statistics
Output summary counts:
- BUY
- SELL
- HOLD

## Story 8 - Unit Tests
Create:
- `tests/test_labeling.py`

Must test:
- future-return calculation
- label assignment
- horizon-shift correctness
- leakage safety

## Story 9 - ML Dataset Schema Contract
Required columns include at least:
- `timestamp`
- `open`, `high`, `low`, `close`, `volume`
- `ema20`, `ema50`, `ema200`
- `rsi`, `macd`, `macd_signal`, `macd_hist`
- `vwap`, `atr`, `bb_width`, `rolling_volume_avg`
- `future_return_5`
- `label`

## Deliverables
Sprint 5 done when repository contains:
- labeling engine
- dataset builder
- export script
- validation checks
- dataset statistics
- unit tests

Output:
- `ml_dataset.csv`
