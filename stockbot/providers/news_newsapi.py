from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import requests

@dataclass
class NewsItem:
    title: str
    url: str
    published_at: str | None
    source: str | None
    sentiment: float  # -1..1 simple heuristic

_POSITIVE = {
    "beat", "beats", "surge", "soar", "soars", "upgrade", "upgrades",
    "raise", "raises", "record", "outperform", "strong", "growth",
}
_NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "downgrade", "downgrades",
    "cut", "cuts", "weak", "slowdown", "slump", "lawsuit",
}


def _sentiment_score(text: str | None) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POSITIVE if w in t)
    neg = sum(1 for w in _NEGATIVE if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, (pos + neg))  # normalize to [-1,1]


def fetch_news_newsapi(ticker: str, api_key: Optional[str], max_items: int = 10) -> List[NewsItem]:
    """Fetch recent news via NewsAPI for a ticker. Returns up to max_items.
    If api_key is None/missing, returns an empty list.
    """
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max(1, min(int(max_items or 0), 25)),
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            return []
        items: List[NewsItem] = []
        for a in data.get("articles", [])[: params["pageSize"]]:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            items.append(
                NewsItem(
                    title=title.strip(),
                    url=a.get("url") or "",
                    published_at=a.get("publishedAt"),
                    source=(a.get("source") or {}).get("name"),
                    sentiment=float(_sentiment_score(title + " " + desc)),
                )
            )
        return items
    except Exception:
        # Degrade gracefully on any error
        return []
