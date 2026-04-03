from __future__ import annotations

import pandas as pd
import pytest

from tradeengine.research.workflow import (
    ResearchWorkflowError,
    build_period_summary,
    build_research_report_template,
    split_research_holdout,
)


def _feature_df() -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2025-01-01T09:15:00+05:30",
        end="2025-12-31T15:15:00+05:30",
        freq="7D",
    )
    rows = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000.0 + i for i in range(rows)],
        }
    )


def test_split_research_holdout_uses_latest_twelve_months() -> None:
    split = split_research_holdout(_feature_df(), research_months=9, holdout_months=3)

    assert len(split.research_months) == 9
    assert len(split.holdout_months) == 3
    assert split.research_months[0] == "2025-01"
    assert split.holdout_months == ["2025-10", "2025-11", "2025-12"]
    assert split.research_df["timestamp"].max() < split.holdout_df["timestamp"].min()


def test_split_research_holdout_rejects_insufficient_months() -> None:
    df = _feature_df()
    with pytest.raises(ResearchWorkflowError, match="Need at least 12 months"):
        split_research_holdout(df[df["timestamp"] < "2025-06-01"], research_months=9, holdout_months=3)


def test_build_period_summary_supports_week_and_month() -> None:
    df = _feature_df()
    weekly = build_period_summary(df, "W")
    monthly = build_period_summary(df, "M")

    assert not weekly.empty
    assert not monthly.empty
    assert {"period", "rows", "start_timestamp", "end_timestamp"} == set(monthly.columns)


def test_build_research_report_template_includes_guardrails() -> None:
    split = split_research_holdout(_feature_df(), research_months=9, holdout_months=3)
    report = build_research_report_template(
        strategy_name="inside_bar_breakout",
        dataset_path="feature_history.csv",
        research_df=split.research_df,
        holdout_df=split.holdout_df,
        research_months=split.research_months,
        holdout_months=split.holdout_months,
    )

    assert "# inside_bar_breakout Research Workflow" in report
    assert "Do not tune on the holdout set." in report
    assert "## Experiment Log" in report
