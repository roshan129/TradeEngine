#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tradeengine.ml.models.evaluation import format_metrics_summary
from tradeengine.ml.models.feature_config import ML_FEATURE_COLUMNS, TARGET_COLUMN
from tradeengine.ml.models.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML classifier from ML dataset CSV")
    parser.add_argument("--dataset", required=True, help="Input ML dataset CSV path")
    parser.add_argument("--output", required=True, help="Output model artifact (.pkl)")
    parser.add_argument("--model-version", default="v1", help="Model version label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    trainer = ModelTrainer(feature_columns=ML_FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    df = trainer.load_dataset(args.dataset)
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
