from __future__ import annotations
import asyncio
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Any, Dict, List

from .config import Settings
from .utils.asyncio_tools import run_with_timeout
from .providers.price_yf import fetch_price_and_fundamentals, fetch_technicals, PriceSnapshot, Technicals
from .providers.news_newsapi import fetch_news_newsapi, NewsItem
from .analysis import recommend

app = FastAPI(title="StockBotVF API", version="0.1.0")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _ds_to_dict(p: PriceSnapshot, t: Technicals, news: List[NewsItem]) -> Dict[str, Any]:
    return {
        "price": {
            "ticker": p.ticker,
            "price": p.price,
            "currency": p.currency,
            "market_cap": p.market_cap,
            "trailing_pe": p.trailing_pe,
            "eps_ttm": p.eps_ttm,
            "dividend_yield": p.dividend_yield,
            "fifty_two_week_high": p.fifty_two_week_high,
            "fifty_two_week_low": p.fifty_two_week_low,
            "earnings_date": p.earnings_date.isoformat() if p.earnings_date else None,
        },
        "technicals": {
            "sma20": t.sma20,
            "sma50": t.sma50,
            "sma200": t.sma200,
            "rsi14": t.rsi14,
            "trend_score": t.trend_score,
        },
        "news": [
            {
                "title": n.title,
                "url": n.url,
                "published_at": n.published_at,
                "source": n.source,
                "sentiment": n.sentiment,
            }
            for n in news
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/analyze")
async def analyze(
    ticker: str = Query(..., description="Ticker symbol, e.g., AAPL"),
    risk: str = Query("medium", pattern="^(low|medium|high)$"),
    max_news: int = Query(10, ge=0, le=25),
    timeout: int = Query(240, ge=10, le=600),
):
    settings = Settings.load()

    async def gather_all():
        p_task = asyncio.create_task(run_with_timeout(asyncio.to_thread(fetch_price_and_fundamentals, ticker), settings.per_request_timeout_seconds))
        t_task = asyncio.create_task(run_with_timeout(asyncio.to_thread(fetch_technicals, ticker), settings.per_request_timeout_seconds))
        n_task = asyncio.create_task(run_with_timeout(asyncio.to_thread(fetch_news_newsapi, ticker, settings.newsapi_key, max_news), settings.per_request_timeout_seconds))
        p = await p_task
        t = await t_task
        news = await n_task
        return p, t, news

    try:
        p, t, news = await run_with_timeout(gather_all(), timeout)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Timed out"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    rec = recommend(p, t, news, risk=risk)

    payload = _ds_to_dict(p, t, news)
    payload["recommendation"] = {
        "label": rec.label,
        "confidence": rec.confidence,
        "rationale": rec.rationale,
    }
    return payload

