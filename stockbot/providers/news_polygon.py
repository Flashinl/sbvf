from __future__ import annotations
from typing import List, Optional
import requests

# Reuse the shared NewsItem shape used elsewhere
from .news_newsapi import NewsItem, _sentiment_score  # type: ignore


def fetch_news_polygon(
    ticker: str,
    api_key: Optional[str],
    max_items: int = 10,
) -> List[NewsItem]:
    """Fetch recent company news via Polygon (if api_key provided).
    Gracefully returns [] on any error.
    Docs: https://polygon.io/docs/stocks/get_v2_reference_news
    """
    if not api_key:
        return []

    try:
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": max(1, min(int(max_items or 0), 25)),
            "apiKey": api_key,
            "order": "desc",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results", []) or []
        out: List[NewsItem] = []
        for a in results:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            desc = a.get("description") or ""
            out.append(
                NewsItem(
                    title=title,
                    url=a.get("article_url") or a.get("url") or "",
                    published_at=a.get("published_utc"),
                    source=((a.get("publisher") or {}).get("name")),
                    sentiment=float(_sentiment_score(title + " " + desc)),
                )
            )
        return out
    except Exception:
        return []

