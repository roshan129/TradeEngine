# VWAP Continuation Hybrid Search

Date: `2026-03-31`

## Scope

Hybrid search over the unlocked VWAP continuation idea using:

- 10 manual logic rounds
- 5 mini-sweep configs inside each round
- 50 configurations per symbol
- 6 months of 5-minute data

Symbols tested:

- `SBIN`
- `ICICIBANK`
- `HDFCBANK`

Base unlocked logic before hybrid rounds:

- EMA filter removed
- VWAP slope filter removed
- candle-direction rule removed
- pullback defined as dip from recent high

## Manual Logic Rounds

The hybrid search varied combinations of:

- tighter entry window (`10:15` vs `10:45`)
- breakout close confirmation
- breakout chase cap
- impulse filter
- volume filter
- stop-cap behavior
- exit mode
- pullback depth band

## Results Summary

### SBIN

- profitable configs found: `5 / 50`
- best result:
  - round: `R6 impulse_filter_06`
  - entry end: `10:15`
  - breakout close confirm: `true`
  - chase cap: `0.15%`
  - impulse filter: `0.6%`
  - exit: `trailing_low`
  - pullback depth: `0.15% -> 0.35%`
  - trades: `5`
  - return: `+0.4613%`
  - profit factor: `2.292`
  - win rate: `40.0%`
  - max drawdown: `0.7105%`
  - expectancy: `92.2513`

Other positive SBIN variants:

- `R6 impulse_filter_06` + `trailing_low` + `0.10% -> 0.25%`
  - return: `+0.3611%`
  - PF: `1.79`
- `R6 impulse_filter_06` + `rr` + `0.10% -> 0.25%`
  - return: `+0.3586%`
  - PF: `1.4618`
- `R6 impulse_filter_06` + `vwap_break` + `0.15% -> 0.35%`
  - return: `+0.2523%`
  - PF: `1.3314`
- `R4 close_confirm_chase_cap` + `vwap_break` + `0.10% -> 0.25%`
  - return: `+0.0359%`
  - PF: `1.0137`

### ICICIBANK

- profitable configs found: `5 / 50`
- best result:
  - round: `R5 impulse_filter_04`
  - entry end: `10:15`
  - breakout close confirm: `true`
  - chase cap: `0.15%`
  - impulse filter: `0.4%`
  - exit: `trailing_low`
  - pullback depth: `0.10% -> 0.25%`
  - trades: `7`
  - return: `+0.2904%`
  - profit factor: `1.4197`
  - win rate: `57.1429%`
  - max drawdown: `0.6839%`
  - expectancy: `41.4893`

Other positive ICICIBANK variants:

- `R9 no_stop_cap_structure_only` + `trailing_low` + `0.10% -> 0.25%`
  - return: `+0.2904%`
  - PF: `1.4197`
- `R10 wider_stop_cap` + `trailing_low` + `0.10% -> 0.25%`
  - return: `+0.2904%`
  - PF: `1.4197`
- `R6 impulse_filter_06` + `vwap_break` + `0.10% -> 0.25%`
  - return: `+0.2652%`
  - PF: `1.4282`
- `R6 impulse_filter_06` + `vwap_break` + `0.15% -> 0.35%`
  - return: `+0.2652%`
  - PF: `1.4282`

### HDFCBANK

- profitable configs found: `1 / 50`
- best result:
  - round: `R7 volume_12`
  - entry end: `10:15`
  - breakout close confirm: `true`
  - chase cap: `0.15%`
  - volume filter: `1.2x`
  - impulse filter: `0.4%`
  - exit: `rr`
  - pullback depth: `0.10% -> 0.25%`
  - trades: `1`
  - return: `+0.5679%`
  - profit factor: `inf`
  - win rate: `100.0%`
  - max drawdown: `0.3430%`
  - expectancy: `567.8948`

Note:

- HDFCBANK result is not reliable yet because it is based on only `1` trade.

## Cross-Symbol Pattern

The strongest recurring pattern across the profitable variants was:

- tighter entry window: `09:20 -> 10:15`
- breakout candle should close above setup high
- chase cap around `0.15%`
- no EMA filter
- no VWAP slope filter
- impulse filter improves quality
- `trailing_low` was the most useful exit for `SBIN` and `ICICIBANK`

## Current Best Candidates

### SBIN Candidate

- round: `R6 impulse_filter_06`
- entry window: `09:20 -> 10:15`
- breakout close confirm: `true`
- chase cap: `0.15%`
- impulse filter: `0.6%`
- exit: `trailing_low`
- pullback depth: `0.15% -> 0.35%`
- stop cap: `0.30%`

### ICICIBANK Candidate

- round: `R5 impulse_filter_04`
- entry window: `09:20 -> 10:15`
- breakout close confirm: `true`
- chase cap: `0.15%`
- impulse filter: `0.4%`
- exit: `trailing_low`
- pullback depth: `0.10% -> 0.25%`
- stop cap: `0.30%`

## Interpretation

This search found promising variants for `SBIN` and `ICICIBANK`, but trade counts are still low.

Recommended next step:

- run out-of-sample validation on the best `SBIN` and `ICICIBANK` candidates
- only convert them into hard presets after they survive that validation

## Runnable Presets

The current best `SBIN` and `ICICIBANK` candidates have now been saved as runnable presets in:

- `scripts/run_vwap_hybrid_preset.py`

Supported preset ids:

- `sbin_best`
- `icici_best`

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_vwap_hybrid_preset.py \
  --preset sbin_best \
  --trades-output data/backtests/sbin_hybrid_preset_trades.csv \
  --equity-output data/backtests/sbin_hybrid_preset_equity.csv
```

The preset runner defaults to the most recent `180` days so it matches this 6-month hybrid-search window unless you override `--lookback-days`.

## SBIN 12-Month Update

Date: `2026-04-01`

A focused 12-month refinement search on `SBIN` improved the earlier candidate.

Window:

- `2025-03-19 13:15 IST -> 2026-03-13 15:25 IST`

New best result:

- experiment cluster: impulse filter in the `0.575% -> 0.585%` band
- official preset value selected: `0.58%`
- entry window: `09:20 -> 10:15`
- breakout close confirm: `true`
- chase cap: `0.15%`
- exit: `rr` at `2R`
- pullback depth: `0.15% -> 0.35%`
- trades: `9`
- return: `+0.8147%`
- profit factor: `1.6348`
- win rate: `66.67%`
- max drawdown: `0.8223%`

Comparison with the previous 12-month SBIN winner:

- previous best: `+0.4568%`, `8` trades, PF `1.3570`
- updated best: `+0.8147%`, `9` trades, PF `1.6348`

Official preset status:

- `sbin_best` in `scripts/run_vwap_hybrid_preset.py` now points to this updated 12-month winner
- the preset now defaults to the trailing `365` days so it reproduces this result directly
