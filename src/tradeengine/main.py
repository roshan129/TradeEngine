from __future__ import annotations

import argparse
import logging

from fastapi import FastAPI

from tradeengine.auth.upstox_auth import UpstoxAuth, UpstoxAuthError, UpstoxCredentials
from tradeengine.api.health import router as health_router
from tradeengine.config import ConfigError, get_upstox_config, settings
from tradeengine.market_data.service import HistoricalDataService
from tradeengine.market_data.upstox_client import UpstoxClient, UpstoxClientError
from tradeengine.utils.logger import configure_logging


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    return app


app = create_app()


def run_historical_data_flow(
    auth_code: str | None = None,
    instrument_key: str | None = None,
    ignore_market_hours: bool = False,
) -> None:
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    try:
        upstox_cfg = get_upstox_config()
    except ConfigError:
        logger.exception("Invalid configuration for Upstox integration")
        raise

    auth = UpstoxAuth(
        credentials=UpstoxCredentials(
            api_key=upstox_cfg.api_key,
            api_secret=upstox_cfg.api_secret,
            redirect_uri=upstox_cfg.redirect_uri,
        )
    )

    if settings.upstox_access_token:
        auth.set_access_token(settings.upstox_access_token)
    else:
        effective_auth_code = auth_code or settings.upstox_auth_code
        if not effective_auth_code:
            login_url = auth.generate_login_url()
            raise UpstoxAuthError(
                "Auth code missing. Open login URL and set UPSTOX_AUTH_CODE or pass --auth-code: "
                f"{login_url}"
            )
        auth.exchange_auth_code_for_access_token(effective_auth_code)

    symbol = instrument_key or settings.upstox_instrument_key
    if not symbol:
        raise ConfigError("Missing UPSTOX_INSTRUMENT_KEY (or pass --instrument-key)")

    service = HistoricalDataService(client=UpstoxClient(auth=auth))
    if ignore_market_hours:
        candles = service.get_last_500_5min_candles_anytime(symbol=symbol)
    else:
        candles = service.get_last_500_5min_candles(symbol=symbol)

    logger.info("Fetched %s five-minute candles for symbol=%s", len(candles), symbol)
    for candle in candles:
        print(
            f"{candle.timestamp.isoformat()} | "
            f"O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeEngine Sprint-1 historical candle fetch runner")
    parser.add_argument("--auth-code", dest="auth_code", help="Upstox authorization code")
    parser.add_argument(
        "--instrument-key",
        dest="instrument_key",
        help="Upstox instrument key (e.g. NSE_EQ|INE002A01018)",
    )
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Fetch candles even when market hours validation would block the request",
    )
    args = parser.parse_args()
    try:
        run_historical_data_flow(
            auth_code=args.auth_code,
            instrument_key=args.instrument_key,
            ignore_market_hours=args.ignore_market_hours,
        )
    except (ConfigError, UpstoxAuthError, UpstoxClientError):
        logging.getLogger(__name__).exception("Historical data flow failed")
