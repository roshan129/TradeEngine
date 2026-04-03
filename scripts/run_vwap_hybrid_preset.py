#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from tradeengine.core.backtester import BacktestConfig, Backtester
from tradeengine.core.strategy import Signal, VwapTrendContinuationStrategy, reverse_signal
from tradeengine.utils.paths import ensure_parent_dir


@dataclass(frozen=True)
class Preset:
    symbol: str
    input_path: str
    lookback_days: int
    entry_end: str
    chase_cap_pct: float
    impulse_pct: float
    exit_mode: str
    pullback_min_pct: float
    pullback_max_pct: float


PRESETS: dict[str, Preset] = {
    "sbin_best": Preset(
        symbol="SBIN",
        input_path="data/market_data/features/feature_history_last_12months_5m_sbin.csv",
        lookback_days=365,
        entry_end="10:15",
        chase_cap_pct=0.0015,
        impulse_pct=0.0058,
        exit_mode="rr",
        pullback_min_pct=0.0015,
        pullback_max_pct=0.0035,
    ),
    "icici_best": Preset(
        symbol="ICICIBANK",
        input_path="data/market_data/features/feature_history_last_12months_5m_icici.csv",
        lookback_days=180,
        entry_end="10:15",
        chase_cap_pct=0.0015,
        impulse_pct=0.004,
        exit_mode="trailing_low",
        pullback_min_pct=0.0010,
        pullback_max_pct=0.0025,
    ),
}


@dataclass
class HybridVwapPresetStrategy(VwapTrendContinuationStrategy):
    require_breakout_close: bool = True
    max_breakout_extension_pct: float = 0.0015
    min_impulse_pct: float = 0.0
    impulse_lookback_bars: int = 5

    def _vwap_slope_ok(self, vwap: float, side: Signal) -> bool:
        del vwap, side
        return True

    def _impulse_ok(self) -> bool:
        if self.min_impulse_pct <= 0:
            return True
        lookback = max(1, min(self.impulse_lookback_bars, len(self._recent_highs), len(self._recent_lows)))
        if len(self._recent_highs) < lookback or len(self._recent_lows) < lookback:
            return False
        recent_high = max(list(self._recent_highs)[-lookback:])
        recent_low = min(list(self._recent_lows)[-lookback:])
        if recent_low <= 0 or recent_high <= recent_low:
            return False
        return ((recent_high - recent_low) / recent_low) >= self.min_impulse_pct

    def _prepare_long_setup(self, row: pd.Series) -> bool:
        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        if any(pd.isna(value) for value in (high, low, close, vwap)):
            return False
        if close <= 0 or vwap <= 0:
            return False
        if close <= vwap or low <= vwap:
            return False
        if not self._recent_strength_ok():
            return False
        if not self._impulse_ok():
            return False

        close_distance_pct = (close - vwap) / vwap
        if close_distance_pct < self.min_distance_above_vwap_pct:
            return False

        pullback_size_pct = self._pullback_size_pct(low)
        if pullback_size_pct is None:
            return False
        if pullback_size_pct < self.min_pullback_size_pct:
            return False
        if pullback_size_pct > self.max_pullback_size_pct:
            return False

        self._pending_setup_high = high
        self._pending_setup_low = low
        return True

    def generate_signal(self, row: pd.Series, context) -> Signal:
        ts = row.get("timestamp")
        if ts is None:
            return "HOLD"
        timestamp = pd.Timestamp(ts)
        if pd.isna(timestamp):
            return "HOLD"

        candle_day = timestamp.date()
        candle_time = timestamp.time()
        if candle_day != self._current_day:
            self._reset_day(candle_day)

        high = float(row.get("high", float("nan")))
        low = float(row.get("low", float("nan")))
        close = float(row.get("close", float("nan")))
        vwap = float(row.get("vwap", float("nan")))
        if pd.isna(high) or pd.isna(low) or pd.isna(close) or pd.isna(vwap):
            return "HOLD"

        signal: Signal = "HOLD"
        try:
            if context.in_position:
                return super().generate_signal(row, context)

            if not self._in_entry_session(candle_time):
                return "HOLD"
            if self._previous_vwap is None:
                return "HOLD"

            if self._pending_setup_high is not None:
                breakout_ok = high > self._pending_setup_high
                if self.require_breakout_close:
                    breakout_ok = breakout_ok and close > self._pending_setup_high
                if breakout_ok and close > vwap and low > vwap:
                    if self.max_breakout_extension_pct > 0 and self._pending_setup_high > 0:
                        extension_pct = (close - self._pending_setup_high) / self._pending_setup_high
                        if extension_pct > self.max_breakout_extension_pct:
                            self._pending_setup_high = None
                            self._pending_setup_low = None
                            return "HOLD"
                    self._pending_stop_loss = self._compute_entry_stop_loss(close)
                    signal = "BUY" if self._pending_stop_loss is not None else "HOLD"
                    self._pending_setup_high = None
                    self._pending_setup_low = None
                    return reverse_signal(signal) if self.reverse_signals else signal

            self._pending_setup_high = None
            self._pending_setup_low = None
            self._prepare_long_setup(row)
            return reverse_signal(signal) if self.reverse_signals else signal
        finally:
            self._finalize_row(close, vwap, high, low)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run saved VWAP hybrid strategy presets")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument("--input", default="", help="Optional override input CSV path")
    parser.add_argument(
        "--trades-output",
        default="data/backtests/vwap_hybrid_preset_trades.csv",
        help="Output CSV path for trade log",
    )
    parser.add_argument(
        "--equity-output",
        default="data/backtests/vwap_hybrid_preset_equity.csv",
        help="Output CSV path for equity curve",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--brokerage-fixed", type=float, default=20.0)
    parser.add_argument("--brokerage-pct", type=float, default=0.0003)
    parser.add_argument("--max-entries-per-day", type=int, default=1)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="Override trailing lookback in days. Use 0 to keep the preset default.",
    )
    parser.add_argument(
        "--stop-after-first-win-per-day",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _parse_hhmm(value: str):
    return pd.Timestamp(f"2026-01-01T{value}:00+05:30").time()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    input_path = args.input or preset.input_path
    lookback_days = args.lookback_days if args.lookback_days > 0 else preset.lookback_days

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    if lookback_days > 0:
        end_ts = pd.Timestamp(df["timestamp"].max())
        start_ts = end_ts - timedelta(days=lookback_days)
        df = df[df["timestamp"] >= start_ts].reset_index(drop=True)

    strategy = HybridVwapPresetStrategy(
        entry_session_start=_parse_hhmm("09:20"),
        entry_session_end=_parse_hhmm(preset.entry_end),
        session_exit_time=_parse_hhmm("15:15"),
        exit_mode=preset.exit_mode,
        risk_reward_multiple=2.0,
        min_candles_above_vwap=2,
        min_distance_above_vwap_pct=0.0010,
        pullback_lookback_bars=5,
        min_pullback_size_pct=preset.pullback_min_pct,
        max_pullback_size_pct=preset.pullback_max_pct,
        fixed_stop_loss_pct=0.003,
        vwap_slope_lookback_bars=1,
        min_vwap_slope_pct=0.0,
        use_ema_trend_filter=False,
        allow_shorts=False,
        require_breakout_close=True,
        max_breakout_extension_pct=preset.chase_cap_pct,
        min_impulse_pct=preset.impulse_pct,
        impulse_lookback_bars=5,
    )
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_atr_multiple=1.0,
        slippage_pct=args.slippage_pct,
        brokerage_fixed=args.brokerage_fixed,
        brokerage_pct=args.brokerage_pct,
        force_end_of_day_exit=True,
        allow_shorts=False,
        max_entries_per_day=max(1, args.max_entries_per_day),
        stop_after_first_win_per_day=bool(args.stop_after_first_win_per_day),
    )

    result = Backtester(strategy=strategy, config=config).run(df)

    ensure_parent_dir(args.trades_output)
    ensure_parent_dir(args.equity_output)
    result.trades.to_csv(args.trades_output, index=False)
    result.equity_curve.to_csv(args.equity_output, index=False)

    print("VWAP Hybrid Preset Result")
    print(f"- Preset: {args.preset}")
    print(f"- Symbol: {preset.symbol}")
    print(f"- Input: {input_path}")
    if lookback_days > 0:
        print(f"- Lookback days: {lookback_days}")
    print(f"- Trades: {len(result.trades)}")
    print(f"- Return %: {result.metrics['total_return_pct']:.4f}")
    print(f"- Win rate %: {result.metrics['win_rate']:.4f}")
    print(f"- Profit factor: {result.metrics['profit_factor']:.4f}")
    print(f"- Max drawdown %: {result.metrics['max_drawdown_pct']:.4f}")
    print(f"- Trades CSV: {args.trades_output}")
    print(f"- Equity CSV: {args.equity_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
