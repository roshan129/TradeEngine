#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from tradeengine.ml.models.evaluation import format_metrics_summary
from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML classifier from ML dataset CSV")
    parser.add_argument("--dataset", required=True, help="Input ML dataset CSV path")
    parser.add_argument("--output", required=True, help="Output model artifact (.pkl)")
    parser.add_argument("--model-version", default="v1", help="Model version label")
    parser.add_argument("--train-session-start", default="09:20")
    parser.add_argument("--train-session-end", default="10:20")
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Allow training with a reduced feature set when columns are missing",
    )
    return parser.parse_args()


def _parse_hhmm(value: str) -> time:
    hour_text, minute_text = value.split(":", 1)
    return time(hour=int(hour_text), minute=int(minute_text))


def _filter_by_session(
    df: pd.DataFrame,
    start: time,
    end: time,
    *,
    label: str,
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        print(f"[WARN] No timestamp column; skipping {label} session filter.")
        return df

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label} data has invalid timestamps; cannot apply session filter.")

    times = timestamps.dt.time
    if start <= end:
        mask = (times >= start) & (times <= end)
    else:
        mask = (times >= start) | (times <= end)

    filtered = df.loc[mask].copy()
    print(
        f"[INFO] {label} session filter {start.strftime('%H:%M')}-{end.strftime('%H:%M')}: "
        f"{len(df)} -> {len(filtered)} rows"
    )
    return filtered


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.dataset)
    session_start = _parse_hhmm(args.train_session_start)
    session_end = _parse_hhmm(args.train_session_end)
    df = _filter_by_session(df, session_start, session_end, label="Training dataset")
    available_features = [col for col in ML_FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in ML_FEATURE_COLUMNS if col not in df.columns]
    if missing_features and not args.allow_missing_features:
        raise ValueError(
            "Missing required feature columns. "
            "Rebuild dataset or pass --allow-missing-features. "
            f"Missing: {', '.join(missing_features)}"
        )
    if missing_features:
        print(
            "[WARN] Missing feature columns; training with reduced set: "
            f"{', '.join(available_features)}"
        )
    if not available_features:
        raise ValueError("No ML feature columns available after filtering.")

    trainer = ModelTrainer(feature_columns=available_features, target_column=TARGET_COLUMN)
    trainer.validate_dataset(df)
    x_train, x_test, y_train, y_test = trainer.split_dataset(df)
    model = trainer.train(x_train, y_train)
    metrics = trainer.evaluate(model, x_test, y_test)
    trainer.save_model(
        model,
        args.output,
        metadata={
            "model_version": args.model_version,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
    )

    print("Model Training Summary")
    print(f"- Training rows: {len(x_train)}")
    print(f"- Test rows: {len(x_test)}")
    for line in format_metrics_summary(metrics):
        print(line)
    print(f"- Model artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
