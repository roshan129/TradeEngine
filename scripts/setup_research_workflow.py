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
        description="Create a disciplined research/holdout workflow from a feature history CSV"
    )
    parser.add_argument("--input", required=True, help="Input feature history CSV path")
    parser.add_argument(
        "--strategy-name",
        default="strategy",
        help="Strategy label used in the generated report",
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
        default="research_workflow",
        help="Directory for split CSVs and markdown report",
    )
    return parser.parse_args()


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

    print("Research Workflow Setup")
    print(f"- Strategy: {args.strategy_name}")
    print(f"- Input CSV: {args.input}")
    print(f"- Research months: {', '.join(split.research_months)}")
    print(f"- Holdout months: {', '.join(split.holdout_months)}")
    print(f"- Research rows: {len(split.research_df)}")
    print(f"- Holdout rows: {len(split.holdout_df)}")
    print(f"- Research CSV: {research_path}")
    print(f"- Holdout CSV: {holdout_path}")
    print(f"- Weekly summary CSV: {weekly_path}")
    print(f"- Monthly summary CSV: {monthly_path}")
    print(f"- Report template: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResearchWorkflowError as exc:
        raise SystemExit(str(exc))
