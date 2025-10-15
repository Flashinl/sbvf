from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List

from .config import Settings
from .providers.price_yf import fetch_price_and_fundamentals, fetch_technicals, PriceSnapshot, Technicals
from .providers.news_newsapi import fetch_news_newsapi, NewsItem
from .analysis import recommend


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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StockBotVF CLI")
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--risk", default="medium", choices=["low", "medium", "high"], help="Risk appetite")
    parser.add_argument("--max-news", type=int, default=8, help="Max news headlines to include (requires NEWSAPI_KEY)")
    args = parser.parse_args(argv)

    settings = Settings.load()

    # Normalize ticker
    ticker = (args.ticker or "").strip().upper()

    # Fetch sequentially (simple CLI flow)
    price = fetch_price_and_fundamentals(ticker)
    tech = fetch_technicals(ticker)
    news = fetch_news_newsapi(
        ticker,
        settings.newsapi_key,
        max_items=max(0, int(args.max_news or 0)),
        company_name=getattr(price, "long_name", None),
        industry=getattr(price, "industry", None),
    )

    rec = recommend(price, tech, news, risk=args.risk)
    payload = _ds_to_dict(price, tech, news)
    payload["recommendation"] = {
        "label": rec.label,
        "confidence": rec.confidence,
        "rationale": rec.rationale,
        "ai_analysis": rec.ai_analysis,
        "predicted_price": rec.predicted_price,
    }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
