#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import pandas as pd

DEFAULT_METRICS = [
    "ema20",
    "ema50",
    "ema200",
    "vwap",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "roc",
    "atr",
    "bb_width",
    "rolling_std",
    "dist_ema20",
    "dist_vwap",
    "rolling_volume_avg",
]


@dataclass(frozen=True)
class MetricPair:
    computed_col: str
    reference_col: str


@dataclass(frozen=True)
class MetricResult:
    name: str
    compared_rows: int
    mae: float
    rmse: float
    max_abs: float
    within_tolerance_pct: float
    passed: bool


def _parse_metric_arg(value: str) -> MetricPair:
    # Accept either "ema20" (same col in both files) or "ema20:EMA 20" (mapped).
    if ":" in value:
        left, right = value.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise argparse.ArgumentTypeError(
                "Invalid --metric value. Use 'computed_col' or 'computed_col:reference_col'."
            )
        return MetricPair(computed_col=left, reference_col=right)

    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Metric cannot be empty")
    return MetricPair(computed_col=value, reference_col=value)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError("Timestamp parsing failed for one or more rows.")
    return ts


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.8f}"


def _evaluate_metric(
    merged: pd.DataFrame,
    pair: MetricPair,
    tolerance: float,
) -> MetricResult:
    comp_col = f"{pair.computed_col}__computed"
    ref_col = f"{pair.reference_col}__reference"

    local = merged[[comp_col, ref_col]].copy()
    local[comp_col] = pd.to_numeric(local[comp_col], errors="coerce")
    local[ref_col] = pd.to_numeric(local[ref_col], errors="coerce")
    local = local.dropna()

    if local.empty:
        return MetricResult(
            name=pair.computed_col,
            compared_rows=0,
            mae=float("nan"),
            rmse=float("nan"),
            max_abs=float("nan"),
            within_tolerance_pct=0.0,
            passed=False,
        )

    diff = (local[comp_col] - local[ref_col]).abs()
    mae = float(diff.mean())
    rmse = float(((local[comp_col] - local[ref_col]) ** 2).mean() ** 0.5)
    max_abs = float(diff.max())
    within_tol = float((diff <= tolerance).mean() * 100.0)

    return MetricResult(
        name=pair.computed_col,
        compared_rows=int(len(local)),
        mae=mae,
        rmse=rmse,
        max_abs=max_abs,
        within_tolerance_pct=within_tol,
        passed=max_abs <= tolerance,
    )


def run_report(
    computed_csv: str,
    reference_csv: str,
    computed_timestamp_col: str,
    reference_timestamp_col: str,
    metrics: list[MetricPair],
    tolerance: float,
) -> tuple[list[MetricResult], int, int]:
    computed_df = pd.read_csv(computed_csv)
    reference_df = pd.read_csv(reference_csv)

    _require_columns(computed_df, [computed_timestamp_col], "Computed CSV")
    _require_columns(reference_df, [reference_timestamp_col], "Reference CSV")
    _require_columns(computed_df, [p.computed_col for p in metrics], "Computed CSV")
    _require_columns(reference_df, [p.reference_col for p in metrics], "Reference CSV")

    computed = computed_df[[computed_timestamp_col, *[p.computed_col for p in metrics]]].copy()
    reference = reference_df[[reference_timestamp_col, *[p.reference_col for p in metrics]]].copy()

    computed[computed_timestamp_col] = _normalize_timestamp(computed[computed_timestamp_col])
    reference[reference_timestamp_col] = _normalize_timestamp(reference[reference_timestamp_col])

    computed = computed.rename(
        columns={
            computed_timestamp_col: "timestamp",
            **{p.computed_col: f"{p.computed_col}__computed" for p in metrics},
        }
    )
    reference = reference.rename(
        columns={
            reference_timestamp_col: "timestamp",
            **{p.reference_col: f"{p.reference_col}__reference" for p in metrics},
        }
    )

    merged = computed.merge(reference, on="timestamp", how="inner")

    results = [_evaluate_metric(merged, pair, tolerance) for pair in metrics]
    return results, len(merged), len(computed_df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare computed feature CSV against a reference CSV and generate indicator error report."
    )
    parser.add_argument("--computed", required=True, help="Path to computed features CSV")
    parser.add_argument("--reference", required=True, help="Path to reference features CSV/chart export")
    parser.add_argument(
        "--computed-timestamp-col",
        default="timestamp",
        help="Timestamp column name in computed CSV",
    )
    parser.add_argument(
        "--reference-timestamp-col",
        default="timestamp",
        help="Timestamp column name in reference CSV",
    )
    parser.add_argument(
        "--metric",
        action="append",
        type=_parse_metric_arg,
        default=[],
        help=(
            "Metric to compare. Repeatable. Format: 'ema20' or 'ema20:EMA 20'. "
            "If omitted, default feature list is used with same names in both files."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Absolute tolerance for pass/fail per metric (default: 0.05)",
    )

    args = parser.parse_args()
    metrics = args.metric or [MetricPair(m, m) for m in DEFAULT_METRICS]

    results, overlap_count, computed_count = run_report(
        computed_csv=args.computed,
        reference_csv=args.reference,
        computed_timestamp_col=args.computed_timestamp_col,
        reference_timestamp_col=args.reference_timestamp_col,
        metrics=metrics,
        tolerance=args.tolerance,
    )

    print("Feature Validation Report")
    print(f"- Computed rows: {computed_count}")
    print(f"- Overlapping timestamps used for compare: {overlap_count}")
    print(f"- Tolerance (abs): {args.tolerance}")
    print("")

    header = f"{'metric':<20} {'rows':>6} {'mae':>14} {'rmse':>14} {'max_abs':>14} {'within_tol_%':>14} {'status':>8}"
    print(header)
    print("-" * len(header))

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            all_passed = False
        print(
            f"{result.name:<20} "
            f"{result.compared_rows:>6} "
            f"{_format_float(result.mae):>14} "
            f"{_format_float(result.rmse):>14} "
            f"{_format_float(result.max_abs):>14} "
            f"{result.within_tolerance_pct:>14.2f} "
            f"{status:>8}"
        )

    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
