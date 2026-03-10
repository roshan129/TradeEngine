# Sprint 6: ML Model Training & Inference Engine

## Goal

Build a deterministic machine learning pipeline that:

1. Trains a predictive model using the dataset built in Sprint 5
2. Evaluates predictive performance
3. Saves a reusable model artifact
4. Provides a prediction interface usable by strategies/backtesting

The model will initially predict:

- BUY
- SELL
- HOLD

based on engineered features.

Later this will evolve into probability-based trading signals.

---

## 1. Architecture Overview

New pipeline layer:

Market Data  
-> Data Cleaning  
-> Feature Engineering  
-> Labeling  
-> Dataset Builder  
-> ML Training (Sprint 6)  
-> Saved Model  
-> Prediction Engine  
-> Strategy / Backtester

Sprint 6 adds two main components:

- Model Trainer
- Model Predictor

---

## 2. New Project Structure

Add a new directory.

`tradeengine/ml/models/`

- `trainer.py`
- `predictor.py`
- `registry.py`
- `evaluation.py`
- `feature_config.py`

Add scripts:

- `scripts/train_model.py`
- `scripts/evaluate_model.py`
- `scripts/predict_from_model.py`

Add tests:

- `tests/test_model_trainer.py`
- `tests/test_model_predictor.py`

---

## 3. Feature Configuration

File:

`tradeengine/ml/models/feature_config.py`

Purpose:
Define exact features used by ML model to prevent accidental feature drift.

Example:

`ML_FEATURE_COLUMNS = ["ema20", "ema50", "ema200", "vwap", "rsi", "macd", "macd_signal", "macd_hist", "roc", "atr", "bb_width", "rolling_std", "dist_ema20", "dist_vwap", "higher_high", "lower_low", "rolling_volume_avg"]`

Also define:

`TARGET_COLUMN = "label"`

---

## 4. Model Trainer

File:

`tradeengine/ml/models/trainer.py`

Class:

`ModelTrainer`

Responsibilities:

- load dataset
- validate schema
- split train/test
- train XGBoost model
- evaluate metrics
- save trained model

Example interface:

```python
class ModelTrainer:
    def __init__(self, feature_columns, target_column):
        ...

    def load_dataset(self, path: str) -> pd.DataFrame:
        ...

    def validate_dataset(self, df: pd.DataFrame):
        ...

    def split_dataset(self, df):
        ...

    def train(self, X_train, y_train):
        ...

    def evaluate(self, model, X_test, y_test):
        ...

    def save_model(self, model, path: str):
        ...
```

---

## 5. Model Algorithm

First model:

`XGBoostClassifier`

Recommended params:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
)
```

Reason:

- fast
- robust
- great for tabular financial data

---

## 6. Dataset Splitting

Do not randomly shuffle financial data.

Use time-based split.

Example:

- 80% training
- 20% testing

Implementation:

```python
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
```

This avoids future data leakage.

---

## 7. Model Evaluation

File:

`tradeengine/ml/models/evaluation.py`

Metrics to compute:

- accuracy
- precision
- recall
- f1 score
- confusion matrix

Also compute:

- class distribution

Example output:

- BUY precision
- SELL precision
- HOLD precision

This helps detect label imbalance problems.

---

## 8. Model Registry

File:

`tradeengine/ml/models/registry.py`

Purpose:

Store trained models with metadata.

Example structure:

```text
models/
  model_v1.pkl
  model_v1_metadata.json
```

Metadata example:

```json
{
  "model_version": "v1",
  "training_date": "2026-03-07",
  "features": ["..."],
  "training_rows": 54000
}
```

---

## 9. Model Predictor

File:

`tradeengine/ml/models/predictor.py`

Class:

`ModelPredictor`

Responsibilities:

- load trained model
- validate feature input
- generate predictions
- return probabilities

Example API:

```python
class ModelPredictor:
    def __init__(self, model_path: str):
        ...

    def predict(self, features: pd.DataFrame):
        ...

    def predict_proba(self, features: pd.DataFrame):
        ...
```

Return example:

- BUY probability: 0.61
- SELL probability: 0.12
- HOLD probability: 0.27

---

## 10. CLI Training Script

File:

`scripts/train_model.py`

Usage:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --dataset ml_dataset.csv \
  --output models/model_v1.pkl
```

Steps:

- load dataset
- train model
- evaluate metrics
- save model
- print metrics

Example output:

- Training rows: 48000
- Test rows: 12000
- Accuracy: 0.58
- BUY precision: 0.62
- SELL precision: 0.55
- HOLD precision: 0.48

---

## 11. Prediction Script

File:

`scripts/predict_from_model.py`

Usage:

```bash
python scripts/predict_from_model.py \
  --model models/model_v1.pkl \
  --input feature_validation_output.csv
```

Output:

- timestamp
- prediction
- buy_probability
- sell_probability
- hold_probability

---

## 12. Backtester Integration (Optional Phase)

Later integrate prediction into strategies.

Example strategy:

`MLSignalStrategy`

Logic:

- if buy_probability > 0.6 -> BUY
- if sell_probability > 0.6 -> SELL
- else HOLD

---

## 13. Tests

Add unit tests:

- `tests/test_model_trainer.py`
- `tests/test_model_predictor.py`

Test cases:

- dataset validation
- feature mismatch
- training runs successfully
- model loads correctly
- prediction shape is valid

---

## 14. Dependencies

Add to `requirements.txt`:

- xgboost
- scikit-learn
- joblib

---

## 15. Deliverables of Sprint 6

After Sprint 6:

- trained ML model
- feature importance analysis
- prediction engine
- model versioning
- evaluation metrics
- CLI training pipeline
