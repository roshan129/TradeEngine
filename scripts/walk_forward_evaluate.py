#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from tradeengine.ml.models.feature_config import TARGET_COLUMN
from tradeengine.ml.models.registry import ModelRegistry
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation for a saved model")
    parser.add_argument("--dataset", required=True, help="Input ML dataset CSV path")
    parser.add_argument("--model", required=True, help="Saved model artifact path")
    parser.add_argument("--initial-train-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--step-fraction", type=float, default=0.1)
    parser.add_argument("--session-start", default="09:20")
    parser.add_argument("--session-end", default="10:20")
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
    bundle = ModelRegistry.load_bundle(args.model)
    trainer = ModelTrainer(
        feature_columns=bundle["feature_columns"],
        target_column=TARGET_COLUMN,
    )
    df = trainer.load_dataset(args.dataset)
    session_start = _parse_hhmm(args.session_start)
    session_end = _parse_hhmm(args.session_end)
    df = _filter_by_session(df, session_start, session_end, label="Walk-forward dataset")
    trainer.validate_dataset(df)

    folds = trainer.walk_forward_indices(
        total_rows=len(df),
        initial_train_fraction=args.initial_train_fraction,
        test_fraction=args.test_fraction,
        step_fraction=args.step_fraction,
    )

    model_template = bundle["model"]
    rows: list[dict[str, float]] = []
    for fold_idx, (train_end, test_end) in enumerate(folds, start=1):
        x_train = trainer._prepare_features(df.iloc[:train_end])  # noqa: SLF001
        y_train = df.iloc[:train_end][TARGET_COLUMN].astype(str)
        x_test = trainer._prepare_features(df.iloc[train_end:test_end])  # noqa: SLF001
        y_test = df.iloc[train_end:test_end][TARGET_COLUMN].astype(str)

        fold_model = model_template.__class__(**model_template.get_xgb_params())
        label_to_int = {label: idx for idx, label in enumerate(trainer.class_labels)}
        y_encoded = y_train.map(label_to_int).astype(int).to_numpy(dtype=int)
        sample_weight = trainer.compute_balanced_sample_weight(y_encoded)
        fold_model.fit(x_train, y_encoded, sample_weight=sample_weight)
        metrics = trainer.evaluate(fold_model, x_test, y_test)

        rows.append(
            {
                "fold": float(fold_idx),
                "train_rows": float(len(x_train)),
                "test_rows": float(len(x_test)),
                "accuracy": float(metrics["accuracy"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
                "buy_recall": float(metrics["by_class"]["BUY"]["recall"]),
                "sell_recall": float(metrics["by_class"]["SELL"]["recall"]),
            }
        )

    out = pd.DataFrame(rows)
    print("Walk-Forward Evaluation Summary")
    print(f"- Folds: {len(out)}")
    print(
        f"- Mean accuracy: {out['accuracy'].mean():.4f}, "
        f"mean F1 macro: {out['f1_macro'].mean():.4f}"
    )
    print(
        f"- Mean BUY recall: {out['buy_recall'].mean():.4f}, "
        f"mean SELL recall: {out['sell_recall'].mean():.4f}"
    )
    print("- Per-fold metrics:")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
