#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from tradeengine.auth.upstox_auth import UpstoxAuth, UpstoxCredentials
from tradeengine.config import get_upstox_config, settings
from tradeengine.core.data_processor import MarketDataProcessor
from tradeengine.core.features import FeatureEngineer
from tradeengine.market_data.models import Candle
from tradeengine.market_data.service import HistoricalDataService
from tradeengine.market_data.upstox_client import UpstoxClient

IST = ZoneInfo("Asia/Kolkata")


def _parse_date(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=IST)


def _candle_to_row(candle: Candle) -> dict[str, object]:
    return {
        "timestamp": candle.timestamp,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


def _build_service() -> HistoricalDataService:
    cfg = get_upstox_config()
    auth = UpstoxAuth(
        UpstoxCredentials(
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            redirect_uri=cfg.redirect_uri,
        )
    )

    token = settings.upstox_access_token
    if not token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is required in .env")
    auth.set_access_token(token)

    return HistoricalDataService(
        client=UpstoxClient(auth=auth),
        enforce_market_hours=False,
    )


def _add_ema9(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    out["ema9"] = out["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    out = out.dropna(subset=["ema9"]).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export stitched historical 1-minute features CSV for backtesting"
    )
    parser.add_argument("--from-date", required=True, help="Start date in IST (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date in IST (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=28,
        help="Days per API request chunk (default: 28; Upstox minute data limit-safe)",
    )
    parser.add_argument(
        "--symbol",
        default="",
        help="Instrument key (defaults to UPSTOX_INSTRUMENT_KEY from .env)",
    )
    parser.add_argument(
        "--output",
        default="feature_history_1m_output.csv",
        help="Output 1-minute features CSV path",
    )
    parser.add_argument(
        "--raw-output",
        default="",
        help="Optional raw stitched 1-minute OHLCV CSV output path",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.2,
        help="Pause between API calls to reduce rate-limit risk",
    )
    args = parser.parse_args()

    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be > 0")

    from_date = _parse_date(args.from_date)
    to_date = _parse_date(args.to_date)
    if to_date < from_date:
        raise ValueError("--to-date must be on or after --from-date")

    symbol = args.symbol or (settings.upstox_instrument_key or "")
    if not symbol:
        raise RuntimeError("Instrument key required via --symbol or UPSTOX_INSTRUMENT_KEY")

    service = _build_service()

    all_candles: list[Candle] = []
    cursor = from_date

    while cursor <= to_date:
        chunk_end = min(cursor + timedelta(days=args.chunk_days - 1), to_date)
        from_ist = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
        to_ist = chunk_end.replace(hour=23, minute=59, second=59, microsecond=0)

        candles = service.get_candles_between(
            symbol=symbol,
            interval="1minute",
            from_ist=from_ist,
            to_ist=to_ist,
        )
        all_candles.extend(candles)

        print(
            f"Fetched {len(candles)} 1m candles for {from_ist.date()} -> {to_ist.date()} "
            f"(running total: {len(all_candles)})"
        )

        cursor = chunk_end + timedelta(days=1)
        if cursor <= to_date and args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    raw_df = pd.DataFrame([_candle_to_row(c) for c in all_candles])

    processor = MarketDataProcessor()
    clean_df = processor.full_clean_pipeline(raw_df, timeframe_minutes=1)

    features_df = FeatureEngineer().full_feature_pipeline(clean_df)
    features_df = _add_ema9(features_df)
    features_df = features_df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    if args.raw_output:
        clean_df.sort_values("timestamp", ascending=True).reset_index(drop=True).to_csv(
            args.raw_output,
            index=False,
        )

    features_df.to_csv(args.output, index=False)

    print(f"Saved 1m features CSV: {args.output} (rows={len(features_df)})")
    if args.raw_output:
        print(f"Saved 1m raw OHLCV CSV: {args.raw_output} (rows={len(clean_df)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
