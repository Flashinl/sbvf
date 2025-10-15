# StockBotVF (MVP)

Real-time(ish) stock analysis CLI that fetches market data, fundamentals, and news in parallel and produces a fast Buy/Hold/Sell recommendation with rationale and source links. Designed to complete within ~5 minutes per ticker (typically seconds) and support any publicly traded stock, subject to provider coverage and rate limits.

## Features (initial)
- Parallel data fetching (price/technicals via Yahoo Finance, news via NewsAPI if key provided)
- Basic fundamentals (market cap, PE, EPS TTM, dividend yield)
- Technicals (20/50/200 SMA trend, RSI(14))
- Earnings date awareness (risk weight if close)
- Naive news sentiment (keyword heuristic; upgradeable to LLM or sentiment API)
- Probabilistic recommendation (Buy/Hold/Sell + confidence)

## Roadmap
- Optional paid providers for true real-time quote/IV/options: Polygon.io, Finnhub, IEX Cloud
- Estimate revisions, short interest, options OI levels, IV/skew
- Sector-relative valuation and historical z-scores
- Web UI (Next.js) and persistence (Supabase)

## Requirements
- Python 3.10+
- API keys (optional, improves data quality/coverage):
  - NEWSAPI_KEY (https://newsapi.org) for headlines
  - FINNHUB_KEY (optional, future)
  - POLYGON_KEY (optional, future)
  - OPENAI_API_KEY (optional, future; for advanced summarization)

## Setup
1) Create and activate a virtual environment (recommended)
2) Install dependencies (ask for permission if running via agent)

```
python -m venv .venv
# Windows
. .venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

3) Configure environment variables (copy .env.example → .env and fill in values)

```
cp .env.example .env
```

## Usage
```
python -m stockbot.main AAPL --horizon 3m --risk medium --benchmark SPY --timeout 240 --max-news 12
```

Outputs an executive summary, technicals, basic fundamentals, recent news, and a probabilistic Buy/Hold/Sell with confidence.

Notes:
- Without API keys, news may be limited; price/fundamentals will use Yahoo Finance (delayed). For strict real-time quotes, plug in a paid provider.
- All fetches run with timeouts and degrade gracefully to ensure completion under your time budget.

## Configuration
- See stockbot/config.py for environment variables.

## Testing (future)
- Unit tests will target analysis scoring and formatting. Run via `pytest` once added.


## Deploying to Render

This repo includes a render.yaml for one-click deployment.

Steps:
1) Push this repo to GitHub/GitLab
2) In Render, create a new Web Service from your repo
3) Render reads render.yaml and configures:
   - Build: pip install -r requirements.txt
   - Start: uvicorn stockbot.api:app --host 0.0.0.0 --port $PORT
4) In Render dashboard, set environment variables (at minimum NEWSAPI_KEY)

API endpoints after deploy:
- GET /health -> {"status":"ok"}
- GET /analyze?ticker=AAPL&risk=medium&max_news=10&timeout=240 -> JSON payload with price, technicals, news, and recommendation

Local dev server:
```
uvicorn stockbot.api:app --reload --port 8000
```
