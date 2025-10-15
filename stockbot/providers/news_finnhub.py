from __future__ import annotations
from typing import List, Optional
from datetime import datetime, timedelta
import requests

# Reuse the shared NewsItem shape used elsewhere
from .news_newsapi import NewsItem, _sentiment_score  # type: ignore


def _iso(ts: int | float | None) -> str | None:
    try:
        if ts is None:
            return None
        return datetime.utcfromtimestamp(int(ts)).isoformat() + "Z"
    except Exception:
        return None


def fetch_news_finnhub(
    ticker: str,
    api_key: Optional[str],
    max_items: int = 10,
) -> List[NewsItem]:
    """Fetch recent company news via Finnhub (if api_key provided).
    Gracefully returns [] on any error.
    Docs: https://finnhub.io/docs/api/company-news
    """
    if not api_key:
        return []

    try:
        today = datetime.utcnow().date()
        frm = (today - timedelta(days=14)).isoformat()
        to = today.isoformat()
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": frm, "to": to, "token": api_key}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        out: List[NewsItem] = []
        for a in (data or [])[: max(1, min(int(max_items or 0), 25))]:
            title = (a.get("headline") or "").strip()
            if not title:
                continue
            desc = a.get("summary") or ""
            out.append(
                NewsItem(
                    title=title,
                    url=a.get("url") or "",
                    published_at=_iso(a.get("datetime")),
                    source=a.get("source"),
                    sentiment=float(_sentiment_score(title + " " + desc)),
                )
            )
        return out
    except Exception:
        return []

