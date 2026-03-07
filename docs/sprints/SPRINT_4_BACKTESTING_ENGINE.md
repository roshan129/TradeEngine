# Sprint 4 - Backtesting Engine

## Sprint Goal
Build a deterministic, event-driven backtesting engine that:
- Simulates trades candle-by-candle
- Applies position sizing
- Applies stop-loss and exit rules
- Accounts for brokerage and slippage
- Tracks capital evolution
- Produces robust performance metrics

No vectorized backtests. One candle at a time.

## High-Level Architecture
Create the following modules:

```text
tradeengine/core/
  strategy.py
  backtester.py
  portfolio.py
  metrics.py
```

Separation of concerns:
- Strategy: signal generation contract
- Backtester: simulation loop and orchestration
- Portfolio: capital, positions, trade lifecycle
- Metrics: performance calculations

## Stories

### Story 1 - Strategy Interface Contract
- Create strategy abstraction with `generate_signal(row, context) -> str`
- Support `BUY`, `SELL`, `HOLD`
- Keep strategy stateless
- Backtester depends only on strategy interface
- Implement one baseline strategy: EMA + RSI

### Story 2 - Baseline Deterministic Strategy
Long-only entry:
- `ema20 > ema50`
- `rsi > 55`
- no open position

Exit conditions:
- Stop loss at 1 ATR (managed by execution engine)
- `rsi < 45`
- End-of-day exit

Acceptance:
- Deterministic signals
- Tested on static dataset

### Story 3 - Portfolio and Position Model
Create models:
- `Position`: entry_price, quantity, stop_loss, entry_time
- `Portfolio`: capital, open_position, trade_log

Responsibilities:
- Enter and exit trades
- Update capital
- Enforce one open position maximum

### Story 4 - Event-Driven Backtest Loop
Loop candle-by-candle:
- generate signal using current row
- enter at current close on BUY if flat
- if position open, check stop and exits

Execution constraints:
- Entry on current candle close
- Stop loss evaluated using low of subsequent candles
- No future leakage

### Story 5 - Position Sizing Engine
- Risk per trade = 1% of current capital
- Stop distance = ATR-based
- Quantity = risk amount / stop distance (bounded by affordability)
- Position size updates as capital changes

### Story 6 - Brokerage and Slippage Modeling
- Brokerage supports fixed and percentage components
- Entry and exit slippage applied to execution price
- Costs deducted from capital

### Story 7 - Trade Logging
Trade record fields:
- entry timestamp, entry price
- exit timestamp, exit price
- quantity
- gross pnl, net pnl
- R multiple
- trade duration
- exit reason

### Story 8 - Performance Metrics Engine
Metrics:
- total return %
- win rate
- average win
- average loss
- profit factor
- max drawdown
- sharpe ratio (basic)
- expectancy

Drawdown calculation is mandatory.

### Story 9 - Equity Curve Generation
Track capital/equity per candle in an equity curve DataFrame for validation and downstream ML workflows.

### Story 10 - Backtest Report CLI
Add `scripts/run_backtest.py` to output:
- summary metrics
- total trades
- max drawdown
- CSV trade log export

## Engineering Principles
1. No lookahead bias
2. No vectorized cheating
3. One candle at a time
4. Deterministic results
5. Separation of strategy and execution logic

## Definition of Done
Sprint 4 is complete when:
- Backtester runs on 3 months historical data
- Produces trade log
- Computes performance metrics
- Includes brokerage and slippage
- Deterministic outputs
- No future leakage
- Unit tests cover core logic
