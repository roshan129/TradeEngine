from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ResearchWorkflowError(ValueError):
    """Raised when a research workflow input or split is invalid."""


@dataclass(frozen=True)
class ResearchSplit:
    research_df: pd.DataFrame
    holdout_df: pd.DataFrame
    split_timestamp: pd.Timestamp
    research_months: list[str]
    holdout_months: list[str]


def _prepare_timestamped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ResearchWorkflowError(f"Expected pandas DataFrame, got: {type(df).__name__}")
    if "timestamp" not in df.columns:
        raise ResearchWorkflowError("Input dataframe must contain a 'timestamp' column")

    out = df.copy(deep=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ResearchWorkflowError("Input dataframe contains invalid timestamp values")

    out = out.sort_values("timestamp", ascending=True).reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ResearchWorkflowError("Input dataframe contains duplicate timestamps")
    return out


def _calendar_periods(timestamp_series: pd.Series, frequency: str) -> pd.Series:
    """Convert timestamps to calendar periods without emitting tz-drop warnings."""
    naive_ts = pd.to_datetime(timestamp_series, errors="coerce").dt.tz_localize(None)
    return naive_ts.dt.to_period(frequency)


def split_research_holdout(
    df: pd.DataFrame,
    *,
    research_months: int = 9,
    holdout_months: int = 3,
) -> ResearchSplit:
    """Split a timestamped dataset into research and untouched holdout windows."""
    if research_months <= 0:
        raise ResearchWorkflowError("research_months must be positive")
    if holdout_months <= 0:
        raise ResearchWorkflowError("holdout_months must be positive")

    clean = _prepare_timestamped_dataframe(df)
    months = _calendar_periods(clean["timestamp"], "M").dropna().unique().tolist()
    ordered_months = sorted(months)
    required_months = research_months + holdout_months
    if len(ordered_months) < required_months:
        raise ResearchWorkflowError(
            f"Need at least {required_months} months of data, found {len(ordered_months)}"
        )

    selected_months = ordered_months[-required_months:]
    research_periods = selected_months[:research_months]
    holdout_periods = selected_months[research_months:]

    research_month_series = _calendar_periods(clean["timestamp"], "M")
    research_mask = research_month_series.isin(research_periods)
    holdout_mask = research_month_series.isin(holdout_periods)
    research_df = clean.loc[research_mask].reset_index(drop=True)
    holdout_df = clean.loc[holdout_mask].reset_index(drop=True)

    if research_df.empty or holdout_df.empty:
        raise ResearchWorkflowError("Research/holdout split produced an empty partition")

    split_timestamp = pd.Timestamp(holdout_df["timestamp"].iloc[0])
    return ResearchSplit(
        research_df=research_df,
        holdout_df=holdout_df,
        split_timestamp=split_timestamp,
        research_months=[str(period) for period in research_periods],
        holdout_months=[str(period) for period in holdout_periods],
    )


def build_period_summary(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Build row-count summaries by calendar period for quick diagnostics."""
    clean = _prepare_timestamped_dataframe(df)
    if frequency not in {"W", "M"}:
        raise ResearchWorkflowError("frequency must be 'W' or 'M'")

    periods = _calendar_periods(clean["timestamp"], frequency).astype(str)
    summary = (
        clean.assign(period=periods)
        .groupby("period", sort=True)
        .agg(
            rows=("timestamp", "count"),
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    return summary


def build_research_report_template(
    *,
    strategy_name: str,
    dataset_path: str,
    research_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    research_months: list[str],
    holdout_months: list[str],
) -> str:
    """Generate a disciplined experiment-log template for one strategy."""
    research_weekly = build_period_summary(research_df, "W")
    research_monthly = build_period_summary(research_df, "M")
    holdout_monthly = build_period_summary(holdout_df, "M")

    weekly_lines = "\n".join(
        f"- `{row.period}`: rows={row.rows}, start={row.start_timestamp}, end={row.end_timestamp}"
        for row in research_weekly.itertuples(index=False)
    )
    monthly_lines = "\n".join(
        f"- `{row.period}`: rows={row.rows}, start={row.start_timestamp}, end={row.end_timestamp}"
        for row in research_monthly.itertuples(index=False)
    )
    holdout_lines = "\n".join(
        f"- `{row.period}`: rows={row.rows}, start={row.start_timestamp}, end={row.end_timestamp}"
        for row in holdout_monthly.itertuples(index=False)
    )

    return f"""# {strategy_name} Research Workflow

## Objective

Improve `{strategy_name}` with a disciplined research loop:

1. baseline on research window only
2. analyze week-by-week and month-by-month behavior
3. change one variable at a time
4. log every experiment
5. run the final candidate once on untouched holdout data

## Dataset Split

- Dataset path: `{dataset_path}`
- Research months ({len(research_months)}): {", ".join(f"`{month}`" for month in research_months)}
- Holdout months ({len(holdout_months)}): {", ".join(f"`{month}`" for month in holdout_months)}
- Research rows: {len(research_df)}
- Holdout rows: {len(holdout_df)}
- Research range: `{research_df["timestamp"].min()}` -> `{research_df["timestamp"].max()}`
- Holdout range: `{holdout_df["timestamp"].min()}` -> `{holdout_df["timestamp"].max()}`

## Guardrails

- Do not tune on the holdout set.
- Do not change multiple rules in one experiment.
- Every experiment must record a hypothesis before running.
- Final holdout evaluation happens only after the research phase yields a stable candidate.

## Baseline

- Strategy version:
- Command:
- Research-window metrics:
- Initial observations:

## Research Window Summary

### Month-by-Month

{monthly_lines}

### Week-by-Week

{weekly_lines}

## Holdout Window Summary

{holdout_lines}

## Experiment Log

| ID | Hypothesis | Single change | Research result | Decision |
| --- | --- | --- | --- | --- |
| EXP-001 |  |  |  |  |

## Current Candidate

- Candidate ID:
- Why this is the current best version:
- Remaining weaknesses:

## Final Holdout Protocol

- Holdout command:
- Run date:
- Holdout result:
- Did it improve vs baseline:
- Conclusion:
"""
