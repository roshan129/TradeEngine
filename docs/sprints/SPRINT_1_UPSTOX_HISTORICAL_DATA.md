TradeEngine - Sprint 1

Historical Market Data Integration (Upstox)

## Sprint Goal

Implement backend capability to fetch historical 5-minute candle data for the last 30 minutes from Upstox API and normalize it into a structured internal format.

This sprint only focuses on:
- Authentication
- API client
- Historical data fetch
- Data normalization
- Logging
- Error handling

No strategy logic.
No database persistence (yet).
No trading logic.

## Technical Stack

- Language: Python 3.11+
- HTTP Client: requests
- Data handling: pandas
- Config management: python-dotenv
- Logging: Python logging
- Project structure: Modular, API-agnostic

## High-Level Architecture

```text
TradeEngine/
|
|- src/
|  |- tradeengine/
|     |- main.py
|     |- config.py
|     |
|     |- auth/
|     |  |- upstox_auth.py
|     |
|     |- market_data/
|     |  |- upstox_client.py
|     |  |- models.py
|     |  |- service.py
|     |
|     |- utils/
|        |- logger.py
|
|- .env
|- requirements.txt
|- SPRINT_1_UPSTOX_HISTORICAL_DATA.md
```

## Sprint Stories (Step-by-Step)

### Story 1 - Project Initialization

Objective

Set up the base Python backend project.

Tasks
- Create virtual environment
- Install dependencies:
  - requests
  - pandas
  - python-dotenv
- Create project folder structure
- Add .env file for secrets
- Add structured logging configuration

Acceptance Criteria
- Project runs without errors
- Logging prints to console
- Environment variables load successfully

### Story 2 - Environment & Configuration Layer

Objective

Centralize configuration management.

Tasks
- Create config.py
- Load:
  - UPSTOX_API_KEY
  - UPSTOX_API_SECRET
  - UPSTOX_REDIRECT_URI
- Validate required variables exist
- Raise error if missing

Acceptance Criteria
- Missing env variables fail fast
- Config module is importable anywhere
- No hardcoded secrets in code

### Story 3 - Authentication Module (OAuth Flow)

Objective

Implement authentication to obtain access token.

Tasks
- Create auth/upstox_auth.py
- Implement:
  - Method to generate login URL
  - Method to exchange auth code for access token
  - Store access token in memory (temporary for v1)
- Handle:
  - Expired token
  - Invalid credentials
  - HTTP failures

Acceptance Criteria
- Can manually authenticate once
- Access token successfully retrieved
- Proper error logging on failure

Note: For automation later, token refresh must be scheduled daily.

### Story 4 - Market Data Client (Raw API Layer)

Objective

Create isolated Upstox client for historical candle API.

Tasks
- Create market_data/upstox_client.py
- Implement method:

```python
fetch_historical_candles(
    instrument_key: str,
    interval: str,
    from_datetime: datetime,
    to_datetime: datetime,
)
```

- Use proper headers:
  - Authorization: Bearer <access_token>
- Handle:
  - 4xx errors
  - 5xx errors
  - Timeout
  - Rate limiting
- Add retry mechanism (basic exponential backoff)

Acceptance Criteria
- Returns raw JSON from API
- Logs failures clearly
- Retries on temporary failures
- No business logic inside this class

### Story 5 - Historical Data Service Layer

Objective

Add abstraction above API client.

Tasks
- Create market_data/service.py
- Implement method:

```python
get_last_30_minutes_5min_candles(symbol: str)
```

Internal logic:
- Calculate:
  - Current IST time
  - Subtract 30 minutes
  - Convert to proper API format
- Call client
- Return structured data

Acceptance Criteria
- Always fetches last 30 minutes dynamically
- Works only during market hours (optional validation)
- Returns normalized data structure

### Story 6 - Data Normalization Layer

Objective

Convert raw API response into structured format.

Tasks
- Create models.py
- Define Candle model:
  - timestamp
  - open
  - high
  - low
  - close
  - volume
- Convert JSON response to:
  - Pandas DataFrame or
  - List of Candle objects
- Ensure:
  - Timestamp converted to IST
  - Sorted ascending
  - No duplicates
  - Numeric fields properly cast

Acceptance Criteria
- Clean structured candle output
- Data verified against actual chart values
- No malformed timestamps
- No null numeric values

### Story 7 - Main Entry Script

Objective

Test end-to-end flow.

Tasks
- Create main.py
- Steps:
  1. Load config
  2. Authenticate
  3. Fetch last 30 min 5-min candles
  4. Print formatted output
  5. Log success

Acceptance Criteria
- Script runs successfully
- Prints structured candle data
- No crashes on failure
- Logs meaningful messages

## Error Handling Requirements

System must:
- Fail fast on invalid config
- Log authentication errors clearly
- Retry on temporary network issues
- Gracefully handle market closed scenarios
- Never crash silently

## Data Validation Checklist

Before moving to next sprint:
- Candle count correct (approx 6 candles for 30 minutes)
- No missing intervals
- Prices match actual chart
- Timezone correct (IST)
- Works multiple times consistently

## Out of Scope (For This Sprint)

- Database storage
- Strategy logic
- Position sizing
- Trading execution
- Sentiment analysis
- Websocket streaming

## Sprint Completion Criteria

Sprint is complete when:
- You can run one command
- System authenticates with Upstox
- System fetches last 30 min 5-min candles
- Data is clean and structured
- No manual patching required
