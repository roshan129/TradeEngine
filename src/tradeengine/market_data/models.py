from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def normalize_candles(raw_payload: dict[str, Any], timezone: str = "Asia/Kolkata") -> list[Candle]:
    candle_rows = raw_payload.get("data", {}).get("candles", [])
    if not isinstance(candle_rows, list) or len(candle_rows) == 0:
        return []

    frame = pd.DataFrame(candle_rows)
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    if frame.shape[1] < len(required_columns):
        return []

    frame = frame.iloc[:, : len(required_columns)]
    frame.columns = required_columns

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(timezone)

    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=required_columns)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.sort_values("timestamp", ascending=True)

    frame["volume"] = frame["volume"].astype(int)

    records: list[Candle] = []
    for row in frame.itertuples(index=False):
        records.append(
            Candle(
                timestamp=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            )
        )
    return records
