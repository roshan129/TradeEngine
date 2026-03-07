# Sprint 5 - Labeling Engine & AI Dataset Builder

## Status
- Completed
- Deterministic labeling + dataset build flow implemented and tested.

## Sprint Goal
Convert engineered feature data into ML-ready supervised datasets:
- features at time `t`
- labels based on forward outcome after `t`

Primary output:
- `ml_dataset.csv`

## Implemented Modules

```text
src/tradeengine/ml/
  __init__.py
  labeling.py
  dataset_builder.py
scripts/
  build_ml_dataset.py
tests/
  test_labeling.py
```

## Story-by-Story Implementation

### Story 1 - Labeling Engine
Implemented in `LabelGenerator` (`src/tradeengine/ml/labeling.py`):
- `generate_labels(...)`
  - uses `close.shift(-horizon)`
  - computes `future_close`, `future_return`
  - label rules:
    - `BUY` if return > buy threshold
    - `SELL` if return < sell threshold
    - else `HOLD`
  - drops rows without future data

### Story 2 - Multi-Horizon Labels
Implemented:
- `generate_multi_horizon_returns(...)`
- Adds:
  - `future_return_5`
  - `future_return_10`
  - `future_return_20`

### Story 3 - Volatility-Adjusted Labels
Implemented:
- `generate_volatility_adjusted_labels(...)`
- Uses ATR-scaled future move threshold (`atr_multiplier * atr`)

### Story 4 - Dataset Builder
Implemented `DatasetBuilder` (`src/tradeengine/ml/dataset_builder.py`):
- Orchestrates multi-horizon returns + label generation
- Supports fixed-threshold and ATR-adjusted label modes
- Drops leakage columns from model input output:
  - `future_close`
  - `future_return`
  - `future_move`

### Story 5 - Dataset Export Script
Implemented script:
- `scripts/build_ml_dataset.py`

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py \
  --input feature_history_output.csv \
  --output ml_dataset.csv
```

### Story 6 - Dataset Validation
Implemented validation contract in `DatasetBuilder`:
- no NaN
- no non-finite numeric values
- sorted timestamps
- unique timestamps

### Story 7 - Dataset Statistics
Implemented:
- `DatasetBuilder.label_counts(...)`
- Outputs BUY/SELL/HOLD counts in CLI summary

### Story 8 - Unit Tests
Implemented:
- `tests/test_labeling.py`

Coverage includes:
- future-return correctness
- label assignment correctness
- horizon shift correctness
- volatility-adjusted labels
- no leakage output columns
- schema checks
- sorted/duplicate timestamp checks

### Story 9 - Schema Contract
Schema contract enforced in `DatasetBuilder.REQUIRED_SCHEMA_COLUMNS`, including:
- OHLCV core
- major engineered indicators
- `future_return_5`
- `label`

## Output Example

```text
timestamp,ema20,ema50,rsi,atr,future_return_5,label
2026-01-02 10:05:00+05:30,101.0,100.0,58.0,1.4,0.004,BUY
```

## CLI Options Implemented

```bash
PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py \
  --input feature_history_output.csv \
  --output ml_dataset.csv \
  --horizons 5,10,20 \
  --label-horizon 5 \
  --buy-threshold 0.003 \
  --sell-threshold -0.003
```

ATR-adjusted labels:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_ml_dataset.py \
  --input feature_history_output.csv \
  --output ml_dataset.csv \
  --use-volatility-adjusted-labels \
  --atr-multiplier 0.5
```

## Deliverables Achieved
- labeling engine
- multi-horizon labels
- volatility-adjusted label mode
- dataset builder
- export CLI
- validation checks
- class-balance stats
- unit tests
- strict schema contract

Output file:
- `ml_dataset.csv`
