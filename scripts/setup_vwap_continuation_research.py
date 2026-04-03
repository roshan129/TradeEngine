#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tradeengine.research.workflow import (
    ResearchWorkflowError,
    build_period_summary,
    build_research_report_template,
    split_research_holdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dedicated research workflow for the VWAP trend continuation strategy"
    )
    parser.add_argument("--input", required=True, help="Input feature history CSV path")
    parser.add_argument(
        "--strategy-name",
        default="vwap_trend_continuation",
        help="Strategy label used in generated reports",
    )
    parser.add_argument(
        "--research-months",
        type=int,
        default=9,
        help="Number of months reserved for research/tuning (default: 9)",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=3,
        help="Number of untouched holdout months (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="research_vwap_trend_continuation",
        help="Directory for split CSVs and markdown reports",
    )
    return parser.parse_args()


def _build_runbook(
    *,
    strategy_name: str,
    research_csv: Path,
    holdout_csv: Path,
    output_dir: Path,
) -> str:
    baseline_command = (
        "PYTHONPATH=src .venv/bin/python scripts/run_backtest.py "
        f"--input {research_csv} "
        f"--strategy {strategy_name} "
        "--vwap-trend-entry-start 09:20 "
        "--vwap-trend-entry-end 10:45 "
        "--vwap-trend-exit-mode vwap_break "
        "--vwap-trend-min-candles-above-vwap 5 "
        "--vwap-trend-min-distance-above-vwap-pct 0.0015 "
        "--vwap-trend-pullback-lookback-bars 5 "
        "--vwap-trend-min-pullback-pct 0.0015 "
        "--vwap-trend-max-pullback-pct 0.0015 "
        "--vwap-trend-fixed-stop-loss-pct 0.003 "
        "--vwap-trend-vwap-slope-lookback-bars 3 "
        "--vwap-trend-min-vwap-slope-pct 0.0001 "
        "--vwap-trend-use-ema-filter "
        "--max-entries-per-day 1 "
        "--stop-after-first-win-per-day "
        f"--trades-output {output_dir / 'baseline_research_trades.csv'} "
        f"--equity-output {output_dir / 'baseline_research_equity.csv'}"
    )
    holdout_command = baseline_command.replace(str(research_csv), str(holdout_csv)).replace(
        "baseline_research_", "holdout_final_"
    )
    grid_command = (
        "PYTHONPATH=src .venv/bin/python scripts/grid_search_vwap.py "
        f"--input {research_csv} "
        f"--output {output_dir / 'vwap_grid_full.csv'} "
        f"--filtered-output {output_dir / 'vwap_grid_filtered.csv'} "
        "--entry-start 09:20 "
        "--entry-end 10:45 "
        "--session-exit 15:15 "
        "--long-only"
    )
    walk_forward_command = (
        "PYTHONPATH=src .venv/bin/python scripts/walk_forward_vwap.py "
        f"--features {research_csv} "
        f"--output-dir {output_dir / 'walk_forward'} "
        "--initial-train-months 6 "
        "--test-months 1 "
        "--step-months 1 "
        "--entry-start 09:20 "
        "--entry-end 10:45 "
        "--session-exit 15:15 "
        "--exit-mode vwap_break "
        "--distance-pct 0.0015 "
        "--min-candles-above-vwap 5 "
        "--pullback-lookback-bars 5 "
        "--min-pullback-pct 0.0015 "
        "--max-pullback-pct 0.0015 "
        "--fixed-stop-loss-pct 0.003 "
        "--vwap-slope-lookback-bars 3 "
        "--min-vwap-slope-pct 0.0001 "
        "--use-ema-filter "
        "--max-trades-per-day 1 "
        "--stop-after-first-win"
    )

    return f"""# VWAP Trend Continuation Runbook

## Goal

Research the long-only VWAP continuation setup on liquid names such as SBI, ICICI, HDFC Bank, and Reliance.

The baseline assumption is:

- price stays above VWAP for multiple candles
- VWAP slope remains positive
- pullback stays above VWAP
- entry happens only on setup-candle breakout
- risk stays tight around the pullback low / 0.3% cap

## Baseline Research Run

```bash
{baseline_command}
```

## Parameter Search

```bash
{grid_command}
```

Search focus:

- `min_candles_above_vwap`: `3, 5, 7`
- `distance_pct`: `0.10%, 0.15%, 0.20%`
- `pullback_pct`: `0.10%, 0.15%, 0.20%`
- `fixed_stop_loss_pct`: `0.25%, 0.30%, 0.35%`
- `exit_mode`: `rr`, `vwap_break`, `trailing_low`

## Walk-Forward Validation

```bash
{walk_forward_command}
```

## Final Holdout Run

Run this only after settling on one candidate:

```bash
{holdout_command}
```

## Notes

- Prefer long-only first. Add shorts only if long-only is stable.
- Keep max entries per day at `1` while researching.
- Use the same settings across SBI and ICICI first before broadening to more names.
"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    split = split_research_holdout(
        df,
        research_months=args.research_months,
        holdout_months=args.holdout_months,
    )

    research_path = output_dir / "research_window.csv"
    holdout_path = output_dir / "holdout_window.csv"
    weekly_path = output_dir / "research_weekly_summary.csv"
    monthly_path = output_dir / "research_monthly_summary.csv"
    report_path = output_dir / "RESEARCH_WORKFLOW.md"
    runbook_path = output_dir / "VWAP_CONTINUATION_RUNBOOK.md"

    split.research_df.to_csv(research_path, index=False)
    split.holdout_df.to_csv(holdout_path, index=False)
    build_period_summary(split.research_df, "W").to_csv(weekly_path, index=False)
    build_period_summary(split.research_df, "M").to_csv(monthly_path, index=False)

    report = build_research_report_template(
        strategy_name=args.strategy_name,
        dataset_path=args.input,
        research_df=split.research_df,
        holdout_df=split.holdout_df,
        research_months=split.research_months,
        holdout_months=split.holdout_months,
    )
    report_path.write_text(report, encoding="utf-8")

    runbook = _build_runbook(
        strategy_name=args.strategy_name,
        research_csv=research_path,
        holdout_csv=holdout_path,
        output_dir=output_dir,
    )
    runbook_path.write_text(runbook, encoding="utf-8")

    print("VWAP Continuation Research Setup")
    print(f"- Strategy: {args.strategy_name}")
    print(f"- Input CSV: {args.input}")
    print(f"- Research months: {', '.join(split.research_months)}")
    print(f"- Holdout months: {', '.join(split.holdout_months)}")
    print(f"- Research CSV: {research_path}")
    print(f"- Holdout CSV: {holdout_path}")
    print(f"- Workflow report: {report_path}")
    print(f"- Runbook: {runbook_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResearchWorkflowError as exc:
        raise SystemExit(str(exc))
