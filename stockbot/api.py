from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from typing import Any, Dict, List, Optional
from pathlib import Path

from .config import Settings
from .utils.asyncio_tools import run_with_timeout
from .providers.price_yf import fetch_price_and_fundamentals, fetch_technicals, PriceSnapshot, Technicals
from .providers.news_newsapi import fetch_news_newsapi, NewsItem
from .providers.news_finnhub import fetch_news_finnhub
from .providers.news_polygon import fetch_news_polygon
from .analysis import recommend
from .auth import router as auth_router, require_user, get_current_user
from .ml_service import init_ml_service, get_ml_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML service on startup"""
    print("[STARTUP] Loading ML model...")
    model_path = Path('models/multihorizon/best_model.pth')
    if model_path.exists():
        try:
            init_ml_service(str(model_path))
            ml = get_ml_service()
            if ml.is_available():
                print("[STARTUP] ML model loaded successfully")
                info = ml.get_model_info()
                print(f"[STARTUP] Model: {info.get('encoder_type', 'unknown')} encoder, "
                      f"{info.get('num_features', 0)} features")
            else:
                print("[STARTUP] ML model not available")
        except Exception as e:
            print(f"[STARTUP] Failed to load ML model: {e}")
            print("[STARTUP] API will run without ML predictions")
    else:
        print("[STARTUP] ML model not found - API will run without ML predictions")

    yield

    print("[SHUTDOWN] Cleaning up...")


app = FastAPI(title="StockBotVF API", version="0.1.0", lifespan=lifespan)
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

    # If ticker looks invalid or no data was found, return a friendly error
    try:
        no_price = (getattr(p, "price", None) is None)
        no_tech = (t is None) or all(getattr(t, fld, None) is None for fld in ("sma20","sma50","sma200","rsi14","trend_score"))
        if no_price and no_tech:
            return JSONResponse(status_code=404, content={"error": "Ticker not found or no data available. Try a valid symbol like AAPL or NVDA."})
    except Exception:
        pass

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

        # Add ML predictions with catalyst detection if model is available
        ml_service = get_ml_service()
        if ml_service.is_available():
            try:
                ml_result = await ml_service.predict_with_explanation(
                    ticker=ticker_norm,
                    news=news,  # Use already-fetched news
                    horizon='1month',
                    fetch_news=False  # Already have news
                )
                if ml_result:
                    payload["ml_prediction"] = {
                        "signal": ml_result['signal'],
                        "expected_return": ml_result['expected_return'],
                        "confidence": ml_result['confidence'],
                        "ml_confidence": ml_result['ml_confidence'],
                        "catalyst_support": ml_result['catalyst_support'],
                        "reasoning": ml_result['reasoning'],
                        "catalysts": ml_result['catalysts'][:3],  # Top 3 catalysts
                        "key_factors": ml_result['key_factors'][:5],  # Top 5 factors
                        "horizons": {
                            horizon: {
                                "signal": pred['signal'],
                                "expected_return": pred['expected_return'],
                                "confidence": pred['confidence']
                            }
                            for horizon, pred in ml_result.get('predictions_all_horizons', {}).items()
                        }
                    }
            except Exception as e:
                print(f"[WARNING] ML prediction failed for {ticker_norm}: {e}")
                payload["ml_prediction"] = None

        return payload
    except Exception as e:
        # Ensure we always return JSON even if recommendation/serialization fails
        return JSONResponse(status_code=500, content={"error": "Analysis failed. Please try another ticker or try again shortly."})


@app.get("/predict/ml")
async def predict_ml(
    user=Depends(require_user),
    ticker: str = Query(..., description="Ticker symbol, e.g., AAPL"),
    horizon: str = Query("1month", pattern="^(1week|1month|3month)$"),
    fetch_news: bool = Query(True, description="Fetch and analyze news catalysts")
):
    """
    ML-based stock prediction with catalyst detection and reasoning

    Returns multi-horizon predictions with:
    - Signal (BUY/HOLD/SELL)
    - Expected return percentage
    - Confidence scores
    - Business catalyst detection (deals, contracts, partnerships, etc.)
    - Human-readable reasoning
    - Key factors driving the prediction
    """
    ml_service = get_ml_service()

    if not ml_service.is_available():
        return JSONResponse(
            status_code=503,
            content={
                "error": "ML prediction service not available",
                "message": "The ML model is not loaded. Please contact support."
            }
        )

    ticker_norm = (ticker or "").strip().upper()

    try:
        result = await ml_service.predict_with_explanation(
            ticker=ticker_norm,
            news=None,
            horizon=horizon,
            fetch_news=fetch_news
        )

        if result is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Prediction failed. Please try again."}
            )

        return {
            "ticker": ticker_norm,
            "horizon": horizon,
            "signal": result['signal'],
            "expected_return": result['expected_return'],
            "confidence": result['confidence'],
            "ml_confidence": result['ml_confidence'],
            "catalyst_support": result['catalyst_support'],
            "reasoning": result['reasoning'],
            "catalysts": result['catalysts'],
            "key_factors": result['key_factors'],
            "all_horizons": result.get('predictions_all_horizons', {})
        }

    except Exception as e:
        print(f"[ERROR] ML prediction failed for {ticker_norm}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed. Please try again."}
        )


@app.get("/predict/ml/batch")
async def predict_ml_batch(
    user=Depends(require_user),
    tickers: str = Query(..., description="Comma-separated ticker symbols, e.g., AAPL,MSFT,GOOGL"),
    horizon: str = Query("1month", pattern="^(1week|1month|3month)$")
):
    """
    Batch ML predictions for multiple tickers

    Returns predictions for all requested tickers
    """
    ml_service = get_ml_service()

    if not ml_service.is_available():
        return JSONResponse(
            status_code=503,
            content={"error": "ML prediction service not available"}
        )

    ticker_list = [t.strip().upper() for t in (tickers or "").split(",") if t.strip()]

    if not ticker_list:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid tickers provided"}
        )

    if len(ticker_list) > 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Maximum 20 tickers allowed per request"}
        )

    try:
        results = {}
        for ticker in ticker_list:
            try:
                result = await ml_service.predict_with_explanation(
                    ticker=ticker,
                    news=None,
                    horizon=horizon,
                    fetch_news=False  # Disable news for batch to save time
                )
                if result:
                    results[ticker] = {
                        "signal": result['signal'],
                        "expected_return": result['expected_return'],
                        "confidence": result['confidence'],
                        "reasoning": result['reasoning']
                    }
            except Exception as e:
                print(f"[WARNING] Prediction failed for {ticker}: {e}")
                results[ticker] = {"error": str(e)}

        return {
            "horizon": horizon,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        print(f"[ERROR] Batch ML prediction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Batch prediction failed"}
        )


@app.get("/ml/model-info")
async def ml_model_info(user=Depends(require_user)):
    """Get information about the loaded ML model"""
    ml_service = get_ml_service()

    if not ml_service.is_available():
        return {
            "available": False,
            "message": "ML model not loaded"
        }

    return ml_service.get_model_info()

