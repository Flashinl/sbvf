from __future__ import annotations
import asyncio
from fastapi import FastAPI, Query, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from typing import Any, Dict, List, Optional

from .config import Settings
from .utils.asyncio_tools import run_with_timeout
from .providers.price_yf import fetch_price_and_fundamentals, fetch_technicals, PriceSnapshot, Technicals
from .providers.news_newsapi import fetch_news_newsapi, NewsItem
from .providers.news_finnhub import fetch_news_finnhub
from .providers.news_polygon import fetch_news_polygon
from .analysis import recommend
from .auth import router as auth_router, require_user, get_current_user

app = FastAPI(title="StockBotVF API", version="0.1.0")
# Session middleware for cookie-based auth
app.add_middleware(SessionMiddleware, secret_key=Settings.load().secret_key, same_site="lax", https_only=False)

# Auth router
app.include_router(auth_router)

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request, user=Depends(get_current_user)):
    if not user:
        return RedirectResponse(url=f"/auth/login?next={request.url.path}", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


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
            "sector": getattr(p, "sector", None),
            "industry": getattr(p, "industry", None),
            "long_name": getattr(p, "long_name", None),
            "business_summary": getattr(p, "long_business_summary", None),
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
    user=Depends(require_user),
    ticker: str = Query(..., description="Ticker symbol, e.g., AAPL"),
    risk: str = Query("medium", pattern="^(low|medium|high)$"),
    max_news: int = Query(10, ge=0, le=25),
    timeout: int = Query(240, ge=10, le=600),
):
    settings = Settings.load()

    # Normalize ticker early
    ticker_norm = (ticker or "").strip().upper()
    # Cap the total request time to stay under Render's gateway timeout
    effective_timeout = min(int(timeout or 60), 90)
    # Keep per-fetch time bounded so all three fetches complete in time
    per_fetch_timeout = min(getattr(settings, "per_request_timeout_seconds", 60), max(15, effective_timeout // 2))

    async def gather_all():
        # Fetch price and technicals in parallel first
        p_task = asyncio.create_task(
            run_with_timeout(asyncio.to_thread(fetch_price_and_fundamentals, ticker_norm), per_fetch_timeout)
        )
        t_task = asyncio.create_task(
            run_with_timeout(asyncio.to_thread(fetch_technicals, ticker_norm), per_fetch_timeout)
        )

        p = await p_task

        # Start multiple news providers in parallel (if keys available)
        news_tasks = []
        # NewsAPI – uses company name/industry for a focused query
        news_tasks.append(
            asyncio.create_task(
                run_with_timeout(
                    asyncio.to_thread(
                        fetch_news_newsapi,
                        ticker_norm,
                        settings.newsapi_key,
                        max_news,
                        getattr(p, "long_name", None),
                        getattr(p, "industry", None),
                    ),
                    per_fetch_timeout,
                )
            )
        )
        if getattr(settings, "finnhub_key", None):
            news_tasks.append(
                asyncio.create_task(
                    run_with_timeout(
                        asyncio.to_thread(
                            fetch_news_finnhub,
                            ticker_norm,
                            settings.finnhub_key,
                            max_news,
                        ),
                        per_fetch_timeout,
                    )
                )
            )
        if getattr(settings, "polygon_key", None):
            news_tasks.append(
                asyncio.create_task(
                    run_with_timeout(
                        asyncio.to_thread(
                            fetch_news_polygon,
                            ticker_norm,
                            settings.polygon_key,
                            max_news,
                        ),
                        per_fetch_timeout,
                    )
                )
            )

        t = await t_task

        # Await all news tasks (degrade gracefully on failures)
        news_lists = []
        for nt in news_tasks:
            try:
                news_lists.append(await nt)
            except Exception:
                news_lists.append([])

        # Merge and dedupe by url/title
        combined = []
        seen = set()
        def _key(n):
            u = (getattr(n, "url", None) or "").strip().lower()
            if u:
                return ("u", u)
            t = (getattr(n, "title", None) or "").strip().lower()
            return ("t", t)

        for lst in news_lists:
            for n in lst or []:
                k = _key(n)
                if not k or k in seen:
                    continue
                seen.add(k)
                combined.append(n)

        news = combined[: max_news]
        return p, t, news

    try:
        p, t, news = await run_with_timeout(gather_all(), effective_timeout)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Timed out"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    try:
        rec = recommend(p, t, news, risk=risk)
        payload = _ds_to_dict(p, t, news)
        payload["recommendation"] = {
            "label": rec.label,
            "confidence": rec.confidence,
            "rationale": rec.rationale,
            "ai_analysis": getattr(rec, "ai_analysis", ""),
            "predicted_price": getattr(rec, "predicted_price", None),
        }
        return payload
    except Exception as e:
        # Ensure we always return JSON even if recommendation/serialization fails
        return JSONResponse(status_code=500, content={"error": str(e)})

