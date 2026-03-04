#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tradeengine.auth.upstox_auth import UpstoxAuth, UpstoxCredentials
from tradeengine.config import get_upstox_config, settings
from tradeengine.core.features import FeatureEngineer
from tradeengine.market_data.service import HistoricalDataService
from tradeengine.market_data.upstox_client import UpstoxClient


def build_feature_dataframe(ignore_market_hours: bool) -> pd.DataFrame:
    """Fetch candles from Upstox and return fully engineered feature dataframe."""
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
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is required in .env for CSV export script")
    auth.set_access_token(token)

    instrument_key = settings.upstox_instrument_key
    if not instrument_key:
        raise RuntimeError("UPSTOX_INSTRUMENT_KEY is required in .env for CSV export script")

    service = HistoricalDataService(
        client=UpstoxClient(auth=auth),
        enforce_market_hours=not ignore_market_hours,
    )

    if ignore_market_hours:
        candles = service.get_last_500_5min_candles_anytime(symbol=instrument_key)
    else:
        candles = service.get_last_500_5min_candles(symbol=instrument_key)

    raw_df = pd.DataFrame(
        [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
    )

    return FeatureEngineer().full_feature_pipeline(raw_df)


def main() -> int:
    """CLI entrypoint: parse args, build features, sort, and write CSV output."""
    parser = argparse.ArgumentParser(description="Export engineered features CSV from Upstox candles")
    parser.add_argument(
        "--output",
        default="feature_validation_output.csv",
        help="Output CSV path (default: feature_validation_output.csv)",
    )
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Sort output timestamps ascending (default is latest-first)",
    )
    parser.add_argument(
        "--respect-market-hours",
        action="store_true",
        help="Use market-hours-gated fetch method (default ignores market-hours)",
    )
    args = parser.parse_args()

    features_df = build_feature_dataframe(ignore_market_hours=not args.respect_market_hours)
    features_df = features_df.sort_values("timestamp", ascending=args.oldest_first).reset_index(drop=True)
    features_df.to_csv(args.output, index=False)

    print(f"Saved {args.output} with {len(features_df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
