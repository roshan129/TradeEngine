# Inside Bar Research Report

Date: 2026-03-23
Instrument: SBI
Primary dataset: 15-minute candles
Research scope: non-ML inside bar breakout strategy

## Goal

Find a workable intraday inside-bar breakout setup for SBI and identify which rules help or hurt.

## Core Questions We Tested

- Mother candle range vs inside bar range
- Volume/VWAP filters on vs off
- Limited morning session vs full-day session
- Daily controls:
  - max 2 trades per day
  - stop after first winning trade of the day
- Risk-reward:
  - 1:1
  - 1:1.5
- Minimum mother-candle range filter
- Long+short vs short-only

## Important Strategy Logic Notes

- Inside bar detection uses candle `high/low`, not `open/close`.
- Default inside-bar breakout uses the mother candle `high/low`.
- Optional mode uses the inside bar candle `high/low`.
- Only one open trade at a time.
- If already in a trade, new inside-bar setups are ignored.
- If nested inside bars appear while a setup is active, the strategy keeps the current active setup and waits for breakout or expiry.
- For the current research default:
  - stop loss comes from mother candle high/low
  - target uses `1.5R`

## What Did Not Work Well

### 1. Strict filter version produced zero trades for one recent week

For SBI on `2026-03-16` to `2026-03-20`, the earlier filtered version produced `0` trades because:

- valid inside bars did exist
- but the breakout candles were blocked by:
  - volume filter
  - VWAP filter
  - or session expiry

This taught us that the default filters were too restrictive for this setup.

### 2. Inside bar range underperformed over larger samples

Using the inside bar candle's own `high/low` for breakout and stop looked okay on a very short slice, but it did not hold up over 3 months.

3-month inside-bar range results:

- `1:1`: return `-3.3805%`, PF `0.8336`
- `1:1.5`: return `-1.1680%`, PF `0.9468`

Conclusion:

- inside-bar range is weaker than mother-candle range on the 3-month sample

### 3. 10:00-13:00 only did not help

6-month test with the current default setup but entry window restricted to `10:00-13:00`:

- return `-2.0929%`
- PF `0.7901`

Conclusion:

- that narrower session did not improve the strategy

## What Worked Better

### 1. Removing volume and VWAP filters helped

When we turned both filters off and used full-day session logic, the strategy finally started taking trades and gave much more informative results.

### 2. Daily controls improved behavior

These rules helped:

- maximum `2` trades per day
- after first winning trade of the day, stop taking new entries

This reduced overtrading and improved the 1-week sample.

### 3. Mother candle range worked better than inside bar range

This became one of the clearest findings.

3-month comparison:

- Mother range, `1:1`: return `-0.5482%`, PF `0.9693`
- Mother range, `1:1.5`: return `-0.2800%`, PF `0.9856`
- Inside range, `1:1`: return `-3.3805%`, PF `0.8336`
- Inside range, `1:1.5`: return `-1.1680%`, PF `0.9468`

Conclusion:

- mother candle breakout range is the better base setup

### 4. Minimum mother-candle range mattered a lot

We swept these values on the 3-month SBI sample using mother-candle range and `1:1.5` RR:

- `0.15%`: return `-0.2800%`, PF `0.9856`
- `0.25%`: return `-0.0653%`, PF `0.9966`
- `0.35%`: return `+2.7635%`, PF `1.1811`
- `0.50%`: return `-0.4988%`, PF `0.9635`

Conclusion:

- `0.35%` minimum mother-candle range was the best tested threshold

## Current Best Research Configuration

This is the current best inside-bar research setup:

- Strategy: `inside_bar_breakout`
- Instrument: SBI
- Timeframe: `15m`
- Breakout range: mother candle `high/low`
- Session: full day `09:15-15:15`
- RR: `1:1.5`
- Minimum mother candle range: `0.35%`
- Volume filter: OFF
- VWAP filter: OFF
- Max trades per day: `2`
- Stop after first winning trade of the day: ON
- Non-ML

This configuration was saved as the current inside-bar research default in:

- `/Users/roshanadke/IdeaProjects/TradeEngine/scripts/run_backtest.py`
- `/Users/roshanadke/IdeaProjects/TradeEngine/README.md`
- `/Users/roshanadke/IdeaProjects/TradeEngine/docs/SCRIPTS.md`

## Important Backtest Results

### 1-week sample (`2026-03-16` to `2026-03-20`)

Mother range with older day rules:

- return `-0.7691%`

Inside-bar range with no filters and day rules:

- return `+0.5865%`

Conclusion:

- inside-bar range looked better on this tiny recent slice
- but it did not generalize on larger samples

### 3-month sample (`2025-12-22` to `2026-03-20`)

Mother range:

- `1:1`: return `-0.5482%`, PF `0.9693`
- `1:1.5`: return `-0.2800%`, PF `0.9856`

After tuning minimum mother range to `0.35%`:

- `1:1.5`: return `+2.7635%`, PF `1.1811`

Conclusion:

- this was the best result we found today

### 6-month sample (`2025-09-22` to `2026-03-20`)

Current default setup:

- return `-2.3271%`
- PF `0.9147`

Conclusion:

- the strategy looked promising over 3 months
- but weakened again over 6 months
- so it is not yet robust enough

## 6-Month Diagnostic Findings

### Month-by-month

Good months:

- `2025-09`: positive
- `2025-12`: positive
- `2026-01`: positive
- `2026-03`: strongly positive

Bad months:

- `2025-10`: strongly negative
- `2025-11`: strongly negative
- `2026-02`: slightly negative

Conclusion:

- strategy is regime-dependent
- two bad months caused most of the 6-month damage

### Long vs short

6-month default run:

- LONG: `52` trades, win rate `38.46%`, net PnL `-4892.92`
- SHORT: `51` trades, win rate `45.10%`, net PnL `+2565.78`

Conclusion:

- shorts are clearly stronger than longs

### Stop-loss damage

6-month default run exit summary:

- `STOP_LOSS`: `30` trades, net PnL `-19131.64`
- `STRATEGY_EXIT_LONG`: `36` trades, net PnL `+4246.77`
- `STRATEGY_EXIT_SHORT`: `37` trades, net PnL `+12557.73`

Conclusion:

- many normal exits are profitable
- the problem is that stop-loss trades are causing most of the damage
- likely meaning some setups are toxic and need to be filtered out

### Entry time tendencies

6-month default run:

- `09:00` hour: weak
- `11:00` hour: strongest
- late afternoon: weak

But our `10:00-13:00` only experiment did not improve overall performance, so this timing clue is not enough by itself.

## Extra Test: Short-Only

6-month short-only version of current default:

- return `-1.4574%`
- PF `0.9210`

Conclusion:

- short-only was better than the full long+short baseline
- but still not profitable

## Best Practical Summary

If we continue tomorrow, the most sensible base to keep testing is:

- SBI
- `15m`
- `inside_bar_breakout`
- mother candle range
- `1:1.5`
- minimum mother-candle range `0.35%`
- no volume filter
- no VWAP filter
- full day
- max `2` trades/day
- stop after first win per day

But we should remember:

- it was good over 3 months
- it was not good enough over 6 months
- shorts look better than longs
- biggest issue is stop-loss damage in bad regimes

## Suggested Next Steps For Tomorrow

- break down stop-loss trades only and look for common patterns
- test more context filters to remove toxic setups
- investigate whether longs should be reduced or removed
- compare bad months (`Oct-Nov`) vs good month (`Mar`) to identify regime differences
